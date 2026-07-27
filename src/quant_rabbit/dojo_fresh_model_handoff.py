"""Fresh-task Codex handoff for paper/replay inventory supervision.

The local compiler is deterministic and uses no model/provider credentials.
It publishes a content-addressed packet only when a model review is useful,
keeps a bounded rolling story instead of conversation history, and accepts
only shadow/research responses that preserve ``order_authority=NONE``.

No function in this module imports a broker client, sends a network request,
starts a process, or applies a recommendation to an account.  Immediate hard
risk guards remain a separate deterministic Python responsibility.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.analysis.market_status import compute_market_status
from quant_rabbit.dojo_paired_inventory_counterfactual import ACTION_IDS, AUTHORITY
from quant_rabbit.dojo_paired_model_queue import (
    canonical_json_bytes,
    canonical_sha256,
)


HANDOFF_PLAN_CONTRACT: Final = "QR_DOJO_FRESH_MODEL_HANDOFF_PLAN_V1"
STORY_CONTRACT: Final = "QR_DOJO_BOUNDED_ROLLING_STORY_V1"
DECISION_PACKET_CONTRACT: Final = "QR_DOJO_FRESH_MODEL_DECISION_PACKET_V1"
MODEL_RESPONSE_CONTRACT: Final = "QR_DOJO_FRESH_MODEL_RESPONSE_V1"
EVENT_CONTRACT: Final = "QR_DOJO_FRESH_MODEL_HANDOFF_EVENT_V1"
STATUS_CONTRACT: Final = "QR_DOJO_FRESH_MODEL_HANDOFF_STATUS_V1"
SKIP_RECEIPT_CONTRACT: Final = "QR_DOJO_FRESH_MODEL_SKIP_RECEIPT_V1"
SCHEMA_VERSION: Final = 1
ZERO_SHA256: Final = "0" * 64
MAX_JSON_BYTES: Final = 2 * 1024 * 1024
MAX_PACKET_BYTES: Final = 256 * 1024
MAX_STORY_ITEMS: Final = 8
MAX_RECENT_EVENTS: Final = 12
MAX_ROOM_EVENT_CANDIDATES: Final = 4
MAX_LEDGER_TAIL_BYTES: Final = 256 * 1024
NORMAL_REVIEW_SECONDS: Final = 60 * 60
HIGH_RISK_REVIEW_SECONDS: Final = 15 * 60
ROOM_FRESHNESS_SECONDS: Final = 15 * 60
MAX_ACTIVE_ROOMS: Final = 32

STORY_FIELDS: Final = (
    "current_thesis",
    "macro_regime",
    "micro_regime",
    "evidence_for",
    "evidence_against",
    "inventory_risk",
    "last_action",
    "expected_outcome",
    "invalidation_conditions",
    "next_review",
    "confidence",
    "known_unknowns",
)
LIST_STORY_FIELDS: Final = {
    "evidence_for",
    "evidence_against",
    "inventory_risk",
    "invalidation_conditions",
    "known_unknowns",
}
TEXT_STORY_FIELDS: Final = {
    "current_thesis",
    "macro_regime",
    "micro_regime",
    "expected_outcome",
    "next_review",
}
RISK_SIGNAL_IDS: Final = {
    "MARGIN_UTILIZATION_HIGH",
    "NET_EXPOSURE_SURGE",
    "GROSS_EXPOSURE_SURGE",
    "CORRELATION_CONCENTRATION",
    "DRAWDOWN_WORSENING",
    "VOLATILITY_REGIME_CHANGE",
    "STRATEGY_THESIS_INVALIDATED",
    "CONSECUTIVE_LOSSES",
    "POSITION_AGE_HIGH",
}
MAJOR_EVENT_IDS: Final = {
    "CENTRAL_BANK_DECISION",
    "INTERVENTION_OR_RATE_CHECK",
    "MARKET_DISLOCATION",
    "MARGIN_GUARD_BREACH",
    "STRATEGY_THESIS_INVALIDATED",
}


class DojoFreshModelHandoffError(ValueError):
    """The snapshot, bounded story, response, or event chain is invalid."""


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DojoFreshModelHandoffError(f"{label} must be lowercase SHA-256")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise DojoFreshModelHandoffError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DojoFreshModelHandoffError(f"{label} must be finite numeric") from exc
    if not math.isfinite(number):
        raise DojoFreshModelHandoffError(f"{label} must be finite numeric")
    return number


def _bounded_text(value: Any, label: str, *, maximum: int = 400) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise DojoFreshModelHandoffError(f"{label} must be bounded text")
    return value


def _bounded_text_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_STORY_ITEMS
        or any(
            not isinstance(item, str) or not item or len(item) > 240 for item in value
        )
    ):
        raise DojoFreshModelHandoffError(f"{label} must be a bounded text list")
    return list(value)


def _read_json(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0 or stat.st_size > maximum_bytes:
        raise DojoFreshModelHandoffError(f"JSON file size is invalid: {path}")
    with resolved.open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DojoFreshModelHandoffError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise DojoFreshModelHandoffError("JSONL tail limit must be positive")
    if path.is_symlink():
        raise DojoFreshModelHandoffError("paper room ledger must not be a symlink")
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            offset = max(0, size - MAX_LEDGER_TAIL_BYTES)
            handle.seek(offset)
            raw = handle.read(MAX_LEDGER_TAIL_BYTES)
    except FileNotFoundError:
        return []
    if offset:
        newline = raw.find(b"\n")
        raw = raw[newline + 1 :] if newline >= 0 else b""
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DojoFreshModelHandoffError("paper room ledger is not UTF-8") from exc
    records = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DojoFreshModelHandoffError(
                "paper room ledger tail is invalid JSONL"
            ) from exc
        if not isinstance(value, dict):
            raise DojoFreshModelHandoffError(
                "paper room ledger record must be an object"
            )
        records.append(value)
    return records


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_or_write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise DojoFreshModelHandoffError(f"immutable artifact conflicts: {path}")
        return
    _write_exclusive(path, value)


def _lock(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        root / ".handoff.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _unlock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def initial_story_content() -> dict[str, Any]:
    """Return the bounded no-history story used before the first model decision."""

    return {
        "current_thesis": "UNESTABLISHED",
        "macro_regime": "UNKNOWN",
        "micro_regime": "UNKNOWN",
        "evidence_for": [],
        "evidence_against": [],
        "inventory_risk": [],
        "last_action": "HOLD",
        "expected_outcome": "NO_PRIOR_MODEL_EXPECTATION",
        "invalidation_conditions": [],
        "next_review": "NORMAL_60M_OR_HIGH_RISK_15M_OR_MAJOR_EVENT",
        "confidence": 0.0,
        "known_unknowns": [
            "MARGIN_UTILIZATION_NOT_PRESENT_UNLESS_SOURCE_SUPPLIES_IT",
            "NO_MODEL_DECISION_HAS_BEEN_ACCEPTED",
        ],
    }


def validate_story_content(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(STORY_FIELDS):
        raise DojoFreshModelHandoffError("rolling story schema mismatch")
    story = dict(value)
    for field in TEXT_STORY_FIELDS:
        _bounded_text(story[field], f"story.{field}")
    for field in LIST_STORY_FIELDS:
        story[field] = _bounded_text_list(story[field], f"story.{field}")
    if story["last_action"] not in ACTION_IDS:
        raise DojoFreshModelHandoffError("story.last_action is outside allowlist")
    confidence = _finite_number(story["confidence"], "story.confidence")
    if confidence < 0 or confidence > 1:
        raise DojoFreshModelHandoffError("story.confidence is outside 0..1")
    story["confidence"] = confidence
    return story


def _build_story_record(
    *,
    story_sequence: int,
    previous_story_sha256: str,
    content: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(story_sequence, int) or story_sequence < 0:
        raise DojoFreshModelHandoffError("story sequence is invalid")
    _sha256_text(previous_story_sha256, "previous story SHA-256")
    body = {
        "contract": STORY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "story_sequence": story_sequence,
        "previous_story_sha256": previous_story_sha256,
        "content": validate_story_content(content),
        "history_policy": "BOUNDED_CURRENT_STORY_ONLY_NO_CONVERSATION_HISTORY",
        "authority": dict(AUTHORITY),
    }
    return {**body, "story_sha256": canonical_sha256(body)}


def verify_story_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    expected_keys = {
        "contract",
        "schema_version",
        "story_sequence",
        "previous_story_sha256",
        "content",
        "history_policy",
        "authority",
        "story_sha256",
    }
    unsigned = {key: item for key, item in record.items() if key != "story_sha256"}
    if (
        set(record) != expected_keys
        or record.get("contract") != STORY_CONTRACT
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("history_policy")
        != "BOUNDED_CURRENT_STORY_ONLY_NO_CONVERSATION_HISTORY"
        or record.get("authority") != dict(AUTHORITY)
        or record.get("story_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoFreshModelHandoffError("rolling story seal is invalid")
    _sha256_text(record["previous_story_sha256"], "previous story SHA-256")
    validate_story_content(record.get("content"))
    return record


def _event_paths(root: Path) -> list[Path]:
    events_dir = root / "events"
    return [] if not events_dir.exists() else sorted(events_dir.glob("*.json"))


def _verify_events(
    root: Path, plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    events: list[dict[str, Any]] = []
    prior = ZERO_SHA256
    for index, path in enumerate(_event_paths(root), start=1):
        row = _read_json(path)
        unsigned = {key: item for key, item in row.items() if key != "event_sha256"}
        expected_name = (
            f"{index:06d}-{row.get('event_type')}-{row.get('event_sha256')}.json"
        )
        if (
            row.get("contract") != EVENT_CONTRACT
            or row.get("schema_version") != SCHEMA_VERSION
            or row.get("handoff_plan_sha256") != plan["handoff_plan_sha256"]
            or row.get("event_index") != index
            or row.get("previous_event_sha256") != prior
            or row.get("event_sha256") != canonical_sha256(unsigned)
            or row.get("authority") != dict(AUTHORITY)
            or path.name != expected_name
        ):
            raise DojoFreshModelHandoffError("handoff event chain is invalid")
        events.append(row)
        prior = row["event_sha256"]
    return events, prior


def _append_event(
    root: Path,
    *,
    plan: Mapping[str, Any],
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    events, prior = _verify_events(root, plan)
    body = {
        "contract": EVENT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "handoff_plan_sha256": plan["handoff_plan_sha256"],
        "event_index": len(events) + 1,
        "previous_event_sha256": prior,
        "event_type": event_type,
        "payload": dict(payload),
        "authority": dict(AUTHORITY),
    }
    row = {**body, "event_sha256": canonical_sha256(body)}
    path = (
        root
        / "events"
        / f"{row['event_index']:06d}-{event_type}-{row['event_sha256']}.json"
    )
    _write_exclusive(path, row)
    return row


def _verify_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(value)
    unsigned = {key: item for key, item in plan.items() if key != "handoff_plan_sha256"}
    if (
        plan.get("contract") != HANDOFF_PLAN_CONTRACT
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("action_allowlist") != list(ACTION_IDS)
        or plan.get("fresh_task_required") is not True
        or plan.get("conversation_history_allowed") is not False
        or plan.get("idle_model_execution_allowed") is not False
        or plan.get("dynamic_event_wake_available") is not False
        or plan.get("authority") != dict(AUTHORITY)
        or plan.get("handoff_plan_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoFreshModelHandoffError("handoff plan is invalid")
    return plan


def initialize_handoff(root: Path) -> dict[str, Any]:
    descriptor = _lock(root)
    try:
        if (root / "handoff-plan.json").exists():
            raise DojoFreshModelHandoffError("handoff is already initialized")
        story = _build_story_record(
            story_sequence=0,
            previous_story_sha256=ZERO_SHA256,
            content=initial_story_content(),
        )
        body = {
            "contract": HANDOFF_PLAN_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "action_allowlist": list(ACTION_IDS),
            "story_fields": list(STORY_FIELDS),
            "story_max_items_per_list": MAX_STORY_ITEMS,
            "recent_event_limit": MAX_RECENT_EVENTS,
            "normal_review_seconds": NORMAL_REVIEW_SECONDS,
            "high_risk_review_seconds": HIGH_RISK_REVIEW_SECONDS,
            "major_event_review": "IMMEDIATE_WHEN_COMPILER_IS_INVOKED",
            "scheduling_runtime": (
                "LOW_FREQUENCY_HEARTBEAT_PLUS_CONTENT_HASH_IDEMPOTENCY"
            ),
            "dynamic_event_wake_available": False,
            "fresh_task_required": True,
            "conversation_history_allowed": False,
            "idle_model_execution_allowed": False,
            "hard_risk_guard_owner": "LOCAL_DETERMINISTIC_PYTHON_NOT_MODEL",
            "model_effect": "SHADOW_RECOMMENDATION_ONLY",
            "cryptographic_response_authority_configured": False,
            "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
            "authority": dict(AUTHORITY),
        }
        plan = {**body, "handoff_plan_sha256": canonical_sha256(body)}
        _write_exclusive(root / "handoff-plan.json", plan)
        _write_exclusive(
            root
            / "stories"
            / f"{story['story_sequence']:06d}-{story['story_sha256']}.json",
            story,
        )
        _append_event(
            root,
            plan=plan,
            event_type="GENESIS",
            payload={"story_sha256": story["story_sha256"]},
        )
        return _status_unlocked(root)
    finally:
        _unlock(descriptor)


def _latest_story(
    root: Path,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    story_sha = str(events[0]["payload"]["story_sha256"])
    sequence = 0
    for event in events[1:]:
        if event["event_type"] == "RESPONSE_ACCEPTED":
            story_sha = str(event["payload"]["next_story_sha256"])
            sequence += 1
    story_path = root / "stories" / f"{sequence:06d}-{story_sha}.json"
    story = verify_story_record(_read_json(story_path))
    if story["story_sequence"] != sequence:
        raise DojoFreshModelHandoffError("story sequence does not match events")
    return story


def _status_unlocked(root: Path) -> dict[str, Any]:
    plan = _verify_plan(_read_json(root / "handoff-plan.json"))
    events, event_tip = _verify_events(root, plan)
    if not events or events[0]["event_type"] != "GENESIS":
        raise DojoFreshModelHandoffError("handoff genesis is missing")
    ready: dict[str, Any] | None = None
    accepted_packet_hashes: set[str] = set()
    last_state_hash: str | None = None
    for event in events[1:]:
        payload = event["payload"]
        if event["event_type"] == "PACKET_READY":
            if ready is not None:
                raise DojoFreshModelHandoffError("multiple packets are ready")
            ready = dict(payload)
        elif event["event_type"] == "RESPONSE_ACCEPTED":
            packet_sha = str(payload.get("decision_packet_sha256") or "")
            if ready is None or ready["decision_packet_sha256"] != packet_sha:
                raise DojoFreshModelHandoffError("response event ordering is invalid")
            accepted_packet_hashes.add(packet_sha)
            last_state_hash = str(ready["decision_state_sha256"])
            ready = None
        else:
            raise DojoFreshModelHandoffError("unsupported handoff event type")
    story = _latest_story(root, events)
    state = "WAITING_FOR_FRESH_TASK" if ready is not None else "IDLE_NO_READY_PACKET"
    body = {
        "contract": STATUS_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "handoff_plan_sha256": plan["handoff_plan_sha256"],
        "state": state,
        "accepted_fresh_model_decision_count": len(accepted_packet_hashes),
        "current_ready_packet_sha256": (
            None if ready is None else ready["decision_packet_sha256"]
        ),
        "current_decision_state_sha256": (
            None if ready is None else ready["decision_state_sha256"]
        ),
        "last_accepted_decision_state_sha256": last_state_hash,
        "current_story_sha256": story["story_sha256"],
        "current_story_sequence": story["story_sequence"],
        "event_count": len(events),
        "event_tip_sha256": event_tip,
        "idle_model_execution_allowed": False,
        "dynamic_event_wake_available": False,
        "cryptographic_provider_signature_verified": False,
        "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "authority": dict(AUTHORITY),
    }
    return {**body, "status_sha256": canonical_sha256(body)}


def handoff_status(root: Path) -> dict[str, Any]:
    descriptor = _lock(root)
    try:
        return _status_unlocked(root)
    finally:
        _unlock(descriptor)


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise DojoFreshModelHandoffError(f"{label} must be ISO-8601 text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoFreshModelHandoffError(f"{label} must be ISO-8601 text") from exc
    if parsed.tzinfo is None:
        raise DojoFreshModelHandoffError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _active_room_contract(
    contract: Mapping[str, Any],
    *,
    now_utc: datetime,
) -> bool:
    if (
        contract.get("contract") != "QR_VIRTUAL_MARKET_SESSION_V2"
        or contract.get("proof_mode") != "diagnostic"
        or contract.get("proof_eligible") is not False
        or contract.get("authority")
        != {
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        }
    ):
        raise DojoFreshModelHandoffError("paper room contract/authority is invalid")
    source = contract.get("source")
    if not isinstance(source, Mapping):
        raise DojoFreshModelHandoffError("paper room source is missing")
    start = _parse_utc(source.get("window_start_utc"), "room window start")
    end = _parse_utc(source.get("window_end_utc"), "room window end")
    if end <= start:
        raise DojoFreshModelHandoffError("paper room window is invalid")
    return start <= now_utc < end


def _room_position_for_source(
    value: Any,
    *,
    room_id: str,
    quotes: Mapping[str, Any],
    now_utc: datetime,
    ceiling_minutes: int,
) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(value, Mapping):
        raise DojoFreshModelHandoffError("room position must be an object")
    pair = str(value.get("pair") or "")
    side = str(value.get("side") or "")
    trade_id = str(value.get("trade_id") or "")
    quote = quotes.get(pair)
    if (
        not pair
        or not trade_id
        or side not in {"LONG", "SHORT"}
        or not isinstance(quote, Mapping)
    ):
        raise DojoFreshModelHandoffError("room position side/quote is invalid")
    entry = _finite_number(value.get("entry_price"), "position.entry_price")
    current = _finite_number(
        quote.get("bid" if side == "LONG" else "ask"),
        "position.current_price",
    )
    pip = 0.01 if pair.endswith("_JPY") else 0.0001
    unrealized_pips = (current - entry) * (1.0 if side == "LONG" else -1.0) / pip
    opened_text = str(value.get("opened_ts") or "")
    opened = _parse_utc(opened_text, "position opened timestamp")
    if (opened - now_utc).total_seconds() > 60:
        raise DojoFreshModelHandoffError("position opened timestamp is in future")
    age_minutes = max(0, int((now_utc - opened).total_seconds() // 60))
    risk_signals = {"POSITION_AGE_HIGH"} if age_minutes >= ceiling_minutes else set()
    return (
        {
            "position_id": f"{room_id}:{trade_id}",
            "room_id": room_id,
            "trade_id": trade_id,
            "pair": pair,
            "side": side,
            "units": value.get("units"),
            "entry_price": entry,
            "current_price": current,
            "unrealized_pips": unrealized_pips,
            "opened_ts": opened_text,
            "strategy_tag": str(value.get("strategy_tag") or ""),
            "tp_price": value.get("tp_price"),
            "sl_price": value.get("sl_price"),
        },
        risk_signals,
    )


def _room_order_for_source(
    value: Any,
    *,
    room_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoFreshModelHandoffError("room order must be an object")
    if value.get("side") not in {"LONG", "SHORT"}:
        raise DojoFreshModelHandoffError("room order side is invalid")
    if not str(value.get("order_id") or "") or not str(value.get("pair") or ""):
        raise DojoFreshModelHandoffError("room order identity is invalid")
    return {
        "room_id": room_id,
        "order_id": str(value.get("order_id") or ""),
        "pair": str(value.get("pair") or ""),
        "side": str(value.get("side") or ""),
        "kind": str(value.get("kind") or ""),
        "limit_price": value.get("limit_price"),
        "units": value.get("units"),
        "strategy_tag": str(value.get("strategy_tag") or ""),
    }


def _room_recent_events_for_source(
    *,
    ledger_path: Path,
    room_id: str,
    now_utc: datetime,
) -> list[dict[str, str]]:
    events = []
    for record in _read_jsonl_tail(
        ledger_path,
        limit=MAX_ROOM_EVENT_CANDIDATES,
    ):
        event_type = _bounded_text(
            str(record.get("event") or ""),
            "room event type",
            maximum=80,
        )
        if not event_type:
            raise DojoFreshModelHandoffError("paper room event type is missing")
        event_time = _parse_utc(record.get("ts_utc"), "room event timestamp")
        if event_time > now_utc:
            raise DojoFreshModelHandoffError(
                "paper room ledger contains a future event"
            )
        event_sha = str(record.get("sha") or "")
        _sha256_text(event_sha, "room event SHA-256")
        payload = record.get("payload")
        payload_map = payload if isinstance(payload, Mapping) else {}
        summary = ";".join(
            (
                f"room={room_id}",
                f"trade={str(payload_map.get('trade_id') or '-')[:48]}",
                f"order={str(payload_map.get('order_id') or '-')[:48]}",
                f"side={str(payload_map.get('side') or '-')[:16]}",
                f"realized_pl_jpy={str(payload_map.get('pl_jpy'))[:32]}",
            )
        )
        events.append(
            {
                "event_id": f"{room_id}:{event_sha}",
                "event_type": event_type,
                "summary": _bounded_text(
                    summary,
                    "room event summary",
                    maximum=240,
                ),
                "available_through_utc": event_time.isoformat(),
            }
        )
    return events


def build_paper_source_packet_from_rooms(
    *,
    rooms_root: Path,
    now_utc: datetime,
) -> tuple[dict[str, Any], list[str]]:
    """Read active local paper rooms without a model or network dependency."""

    if now_utc.tzinfo is None:
        raise DojoFreshModelHandoffError("compiler time must be timezone-aware")
    now = now_utc.astimezone(timezone.utc)
    resolved_root = rooms_root.resolve(strict=True)
    if not resolved_root.is_dir() or rooms_root.is_symlink():
        raise DojoFreshModelHandoffError("rooms root must be a real directory")
    contract_paths = sorted(resolved_root.glob("*/*/session_contract.json"))
    active: list[tuple[Path, dict[str, Any]]] = []
    for contract_path in contract_paths:
        if contract_path.is_symlink() or contract_path.parent.is_symlink():
            raise DojoFreshModelHandoffError("paper room path must not be a symlink")
        contract = _read_json(contract_path)
        if _active_room_contract(contract, now_utc=now):
            active.append((contract_path.parent, contract))
    if not active or len(active) > MAX_ACTIVE_ROOMS:
        raise DojoFreshModelHandoffError("active paper room count must be 1..32")

    rooms: list[dict[str, Any]] = []
    feature_candidates: dict[str, list[dict[str, Any]]] = {}
    risk_signals: set[str] = set()
    recent_event_candidates: list[dict[str, str]] = []
    for room_dir, contract in active:
        room_id = str(contract.get("room_id") or "")
        if not room_id or room_dir.name != room_id:
            raise DojoFreshModelHandoffError("paper room directory/identity mismatch")
        snapshot = _read_json(room_dir / "broker_snapshot.json")
        state = _read_json(room_dir / "state.json")
        wall_time = _parse_utc(state.get("wall_time_utc"), "room wall time")
        age_seconds = (now - wall_time).total_seconds()
        if age_seconds < -60 or age_seconds > ROOM_FRESHNESS_SECONDS:
            raise DojoFreshModelHandoffError(f"active paper room is stale: {room_id}")
        raw_positions = snapshot.get("positions")
        raw_orders = snapshot.get("orders")
        quotes = state.get("quotes")
        account = state.get("account")
        if (
            not isinstance(raw_positions, list)
            or not isinstance(raw_orders, list)
            or not isinstance(quotes, Mapping)
            or not isinstance(account, Mapping)
        ):
            raise DojoFreshModelHandoffError(
                "paper room snapshot/state schema is invalid"
            )
        margin_usage = _finite_number(account.get("margin_usage"), "room margin usage")
        if margin_usage < 0:
            raise DojoFreshModelHandoffError("room margin usage must be nonnegative")
        if margin_usage >= 0.5:
            risk_signals.add("MARGIN_UTILIZATION_HIGH")
        bot = contract.get("bot")
        if not isinstance(bot, Mapping):
            raise DojoFreshModelHandoffError("paper room bot contract is missing")
        config = bot.get("config")
        if not isinstance(config, Mapping):
            raise DojoFreshModelHandoffError("paper room bot config is missing")
        ceiling_minutes = int(config.get("ceiling_min") or 60)
        if ceiling_minutes <= 0:
            raise DojoFreshModelHandoffError("position ceiling must be positive")
        positions = []
        for raw in raw_positions:
            position, signals = _room_position_for_source(
                raw,
                room_id=room_id,
                quotes=quotes,
                now_utc=now,
                ceiling_minutes=ceiling_minutes,
            )
            positions.append(position)
            risk_signals.update(signals)
        orders = [_room_order_for_source(raw, room_id=room_id) for raw in raw_orders]
        recent_event_candidates.extend(
            _room_recent_events_for_source(
                ledger_path=room_dir / "ledger.jsonl",
                room_id=room_id,
                now_utc=now,
            )
        )
        configured_pairs = [
            str(item) for item in config.get("pairs") or contract.get("pairs") or []
        ]
        for pair, raw_quote in quotes.items():
            if not isinstance(raw_quote, Mapping):
                raise DojoFreshModelHandoffError("room quote must be an object")
            quote_time = _parse_utc(raw_quote.get("ts"), "room quote timestamp")
            stale_limit = _finite_number(
                contract["source"].get("stale_quote_max_seconds", 90.0),
                "stale quote limit",
            )
            quote_age = (now - quote_time).total_seconds()
            if quote_age < -60 or quote_age > stale_limit:
                raise DojoFreshModelHandoffError("paper room quote is stale")
            bid = _finite_number(raw_quote.get("bid"), "room quote bid")
            ask = _finite_number(raw_quote.get("ask"), "room quote ask")
            if ask < bid:
                raise DojoFreshModelHandoffError("room quote is crossed")
            feature_candidates.setdefault(str(pair), []).append(
                {
                    "quote_time": quote_time,
                    "feature": {
                        "pair": str(pair),
                        "last_mid": (bid + ask) / 2.0,
                        "price_component": "BID_ASK_SNAPSHOT",
                        "source": "LOCAL_PAPER_ROOM_STATE",
                    },
                }
            )
        rooms.append(
            {
                "experiment_id": str(contract.get("experiment_id") or ""),
                "room_id": room_id,
                "candidate_id": str(contract.get("candidate_id") or ""),
                "strategy_tags": [str(item) for item in bot.get("strategy_tags") or []],
                "configured_pairs": configured_pairs,
                "balance_jpy": snapshot.get("balance_jpy"),
                "positions": positions,
                "orders": orders,
                "state_wall_time_utc": wall_time.isoformat(),
                "state_age_seconds": round(age_seconds, 3),
                "ledger_tip_sha256": str(snapshot.get("ledger_sha") or ""),
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
            }
        )
    market_features = {
        pair: max(rows, key=lambda item: item["quote_time"])["feature"]
        for pair, rows in sorted(feature_candidates.items())
    }
    market_status = compute_market_status(now).to_dict()
    body = {
        "contract": "QR_DOJO_PAPER_AI_SHADOW_HOURLY_V1",
        "generated_at_utc": now.isoformat(),
        "purpose": "SHADOW_ONLY_INVENTORY_AND_MARKET_STORY_REVIEW",
        "market_status": market_status,
        "market_features": market_features,
        "rooms": sorted(rooms, key=lambda item: item["room_id"]),
        "recent_events": sorted(
            recent_event_candidates,
            key=lambda item: (
                item["available_through_utc"],
                item["event_id"],
            ),
        )[-MAX_RECENT_EVENTS:],
        "local_compiler": {
            "network_access_used": False,
            "model_credentials_used": False,
            "broker_client_used": False,
            "source": "LOCAL_PAPER_ROOM_STATE_AND_SNAPSHOT",
        },
        "safety": {
            "paper_only": True,
            "shadow_only": True,
            "order_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
            "recommendations_are_not_commands": True,
        },
    }
    return (
        {**body, "packet_sha256": canonical_sha256(body)},
        sorted(risk_signals),
    )


def _normalize_position(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoFreshModelHandoffError("position row must be an object")
    row = dict(value)
    keys = {
        "position_id",
        "room_id",
        "trade_id",
        "pair",
        "side",
        "units",
        "entry_price",
        "current_price",
        "unrealized_pips",
        "strategy_tag",
        "tp_price",
        "sl_price",
    }
    normalized = {key: row.get(key) for key in sorted(keys)}
    if (
        not isinstance(normalized["position_id"], str)
        or not normalized["position_id"]
        or not normalized["trade_id"]
        or not normalized["pair"]
        or normalized["side"] not in {"LONG", "SHORT"}
    ):
        raise DojoFreshModelHandoffError("position identity/side is invalid")
    for field in ("units", "entry_price"):
        if _finite_number(normalized[field], f"position.{field}") <= 0:
            raise DojoFreshModelHandoffError(f"position.{field} must be positive")
    for field in ("current_price", "unrealized_pips", "tp_price", "sl_price"):
        if normalized[field] is not None:
            number = _finite_number(normalized[field], f"position.{field}")
            if field != "unrealized_pips" and number <= 0:
                raise DojoFreshModelHandoffError(f"position.{field} must be positive")
    return normalized


def _normalize_order(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoFreshModelHandoffError("order row must be an object")
    row = dict(value)
    keys = {
        "room_id",
        "order_id",
        "pair",
        "side",
        "kind",
        "limit_price",
        "units",
        "strategy_tag",
    }
    normalized = {key: row.get(key) for key in sorted(keys)}
    if (
        not isinstance(normalized["order_id"], str)
        or not normalized["order_id"]
        or not normalized["pair"]
        or normalized["side"] not in {"LONG", "SHORT"}
    ):
        raise DojoFreshModelHandoffError("order identity/side is invalid")
    for field in ("limit_price", "units"):
        if normalized[field] is not None:
            if _finite_number(normalized[field], f"order.{field}") <= 0:
                raise DojoFreshModelHandoffError(f"order.{field} must be positive")
    return normalized


def _normalize_source_snapshot(
    source: Mapping[str, Any],
    *,
    risk_signals: Sequence[str],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], datetime]:
    safety = source.get("safety")
    rooms = source.get("rooms")
    market_status = source.get("market_status")
    market_features = source.get("market_features")
    if (
        source.get("contract") != "QR_DOJO_PAPER_AI_SHADOW_HOURLY_V1"
        or not isinstance(safety, Mapping)
        or safety.get("paper_only") is not True
        or safety.get("shadow_only") is not True
        or safety.get("order_authority") != "NONE"
        or safety.get("live_permission") is not False
        or safety.get("broker_mutation_allowed") is not False
        or not isinstance(rooms, list)
        or not isinstance(market_status, Mapping)
        or not isinstance(market_features, Mapping)
    ):
        raise DojoFreshModelHandoffError("source paper safety/schema is invalid")
    source_packet_sha = source.get("packet_sha256")
    if not isinstance(source_packet_sha, str):
        raise DojoFreshModelHandoffError("source packet has no content seal")
    unsigned = {key: item for key, item in source.items() if key != "packet_sha256"}
    if source_packet_sha != canonical_sha256(unsigned):
        raise DojoFreshModelHandoffError("source packet content seal is invalid")
    source_cutoff = _parse_utc(
        source.get("generated_at_utc"),
        "source packet generated timestamp",
    )
    source_recent_events = _validate_recent_events(
        source.get("recent_events") or [],
        cutoff_utc=source_cutoff,
    )
    normalized_rooms = []
    for value in rooms:
        if not isinstance(value, Mapping):
            raise DojoFreshModelHandoffError("room row must be an object")
        room = dict(value)
        if (
            room.get("paper_only") is not True
            or room.get("live_permission") is not False
            or room.get("order_authority") != "NONE"
            or not isinstance(room.get("positions"), list)
            or not isinstance(room.get("orders"), list)
        ):
            raise DojoFreshModelHandoffError("room authority/schema is invalid")
        normalized_rooms.append(
            {
                "balance_jpy": _finite_number(
                    room.get("balance_jpy"), "room.balance_jpy"
                ),
                "candidate_id": str(room.get("candidate_id") or ""),
                "configured_pairs": sorted(
                    str(item) for item in room.get("configured_pairs") or []
                ),
                "experiment_id": str(room.get("experiment_id") or ""),
                "ledger_tip_sha256": str(room.get("ledger_tip_sha256") or ""),
                "orders": sorted(
                    (_normalize_order(item) for item in room["orders"]),
                    key=lambda item: item["order_id"],
                ),
                "positions": sorted(
                    (_normalize_position(item) for item in room["positions"]),
                    key=lambda item: item["position_id"],
                ),
                "room_id": str(room.get("room_id") or ""),
                "strategy_tags": sorted(
                    str(item) for item in room.get("strategy_tags") or []
                ),
            }
        )
        if not normalized_rooms[-1]["room_id"]:
            raise DojoFreshModelHandoffError("room identity is missing")
        _sha256_text(
            normalized_rooms[-1]["ledger_tip_sha256"],
            "room ledger tip SHA-256",
        )
    normalized_rooms.sort(key=lambda item: item["room_id"])
    market = {
        "active_sessions": sorted(
            str(item) for item in market_status.get("active_sessions") or []
        ),
        "closed_reason": market_status.get("closed_reason"),
        "is_fx_open": market_status.get("is_fx_open"),
        "most_recent_open_utc": market_status.get("most_recent_open_utc"),
    }
    if not isinstance(market["is_fx_open"], bool):
        raise DojoFreshModelHandoffError("market open state must be boolean")
    features = {}
    for pair, value in sorted(market_features.items()):
        if not isinstance(value, Mapping):
            raise DojoFreshModelHandoffError("market feature row must be object")
        features[str(pair)] = {
            key: item
            for key, item in sorted(value.items())
            if key
            in {
                "pair",
                "last_complete_m1_utc",
                "last_mid",
                "return_1h_pips",
                "return_4h_pips",
                "return_24h_pips",
                "mean_m1_range_1h_pips",
                "complete_m1_count",
                "price_component",
                "source",
            }
        }
    signals = sorted(set(risk_signals))
    if any(signal not in RISK_SIGNAL_IDS for signal in signals):
        raise DojoFreshModelHandoffError("unknown deterministic risk signal")
    snapshot = {
        "market_status": market,
        "market_features": features,
        "rooms": normalized_rooms,
        "deterministic_risk_signals": signals,
        "future_information_visible": False,
        "terminal_outcome_visible": False,
        "append_wall_clock_visible": False,
    }
    return snapshot, source_packet_sha, source_recent_events, source_cutoff


def _validate_recent_events(
    value: Sequence[Mapping[str, Any]],
    *,
    cutoff_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    if len(value) > MAX_RECENT_EVENTS:
        raise DojoFreshModelHandoffError("recent event window is too large")
    events = []
    event_ids: set[str] = set()
    expected = {"event_id", "event_type", "summary", "available_through_utc"}
    for item in value:
        if not isinstance(item, Mapping) or set(item) != expected:
            raise DojoFreshModelHandoffError("recent event schema mismatch")
        event = dict(item)
        for field in expected:
            _bounded_text(event[field], f"recent_event.{field}", maximum=240)
        if not event["event_id"] or event["event_id"] in event_ids:
            raise DojoFreshModelHandoffError("recent event identity is invalid")
        event_ids.add(event["event_id"])
        available_through = _parse_utc(
            event["available_through_utc"],
            "recent event available-through timestamp",
        )
        if cutoff_utc is not None and available_through > cutoff_utc:
            raise DojoFreshModelHandoffError(
                "recent event is later than the causal cutoff"
            )
        forbidden = json.dumps(event, ensure_ascii=False).lower()
        if any(
            word in forbidden
            for word in ("post_outcome", "terminal_result", "future_quote")
        ):
            raise DojoFreshModelHandoffError(
                "recent event contains forbidden outcome data"
            )
        events.append(event)
    return events


def _previous_decision_summary(
    root: Path, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event["event_type"] != "RESPONSE_ACCEPTED":
            continue
        packet_sha = str(event["payload"]["decision_packet_sha256"])
        response = verify_model_response(
            _read_json(root / "responses" / f"{packet_sha}.json"),
            verify_decision_packet(_read_json(root / "packets" / f"{packet_sha}.json")),
        )
        return {
            "decision_packet_sha256": packet_sha,
            "action": response["action"],
            "reason_ids": response["reason_ids"],
            "next_story_sha256": event["payload"]["next_story_sha256"],
        }
    return None


def _packet_body(
    *,
    plan: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    source_packet_sha256: str,
    story: Mapping[str, Any],
    previous_decision: Mapping[str, Any] | None,
    recent_events: Sequence[Mapping[str, Any]],
    cadence_mode: str,
    decision_state_sha256: str,
) -> dict[str, Any]:
    return {
        "contract": DECISION_PACKET_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "handoff_plan_sha256": plan["handoff_plan_sha256"],
        "source_packet_sha256": source_packet_sha256,
        "decision_state_sha256": decision_state_sha256,
        "snapshot": dict(snapshot),
        "rolling_story": dict(story),
        "previous_decision": (
            None if previous_decision is None else dict(previous_decision)
        ),
        "recent_events": [dict(item) for item in recent_events],
        "cadence_mode": cadence_mode,
        "action_allowlist": list(ACTION_IDS),
        "read_scope": [
            "CURRENT_SNAPSHOT",
            "PREVIOUS_DECISION_SUMMARY",
            "CURRENT_BOUNDED_ROLLING_STORY",
            "BOUNDED_RECENT_EVENTS",
            "STATIC_ACTION_AND_SAFETY_CONTRACT",
        ],
        "conversation_history_allowed": False,
        "fresh_codex_task_required": True,
        "future_information_allowed": False,
        "terminal_outcome_allowed": False,
        "append_wall_clock_allowed": False,
        "hard_risk_guard_owner": "LOCAL_DETERMINISTIC_PYTHON_NOT_MODEL",
        "model_effect": "SHADOW_RECOMMENDATION_ONLY",
        "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "authority": dict(AUTHORITY),
    }


def verify_decision_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    packet = dict(value)
    expected_keys = set(
        _packet_body(
            plan={"handoff_plan_sha256": ZERO_SHA256},
            snapshot={},
            source_packet_sha256=ZERO_SHA256,
            story={},
            previous_decision=None,
            recent_events=[],
            cadence_mode="NORMAL_60M",
            decision_state_sha256=ZERO_SHA256,
        )
    ) | {"decision_packet_sha256"}
    unsigned = {
        key: item for key, item in packet.items() if key != "decision_packet_sha256"
    }
    if (
        set(packet) != expected_keys
        or packet.get("contract") != DECISION_PACKET_CONTRACT
        or packet.get("schema_version") != SCHEMA_VERSION
        or packet.get("action_allowlist") != list(ACTION_IDS)
        or packet.get("read_scope")
        != [
            "CURRENT_SNAPSHOT",
            "PREVIOUS_DECISION_SUMMARY",
            "CURRENT_BOUNDED_ROLLING_STORY",
            "BOUNDED_RECENT_EVENTS",
            "STATIC_ACTION_AND_SAFETY_CONTRACT",
        ]
        or packet.get("conversation_history_allowed") is not False
        or packet.get("fresh_codex_task_required") is not True
        or packet.get("future_information_allowed") is not False
        or packet.get("terminal_outcome_allowed") is not False
        or packet.get("append_wall_clock_allowed") is not False
        or packet.get("model_effect") != "SHADOW_RECOMMENDATION_ONLY"
        or packet.get("authority") != dict(AUTHORITY)
        or packet.get("decision_packet_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoFreshModelHandoffError("fresh-task decision packet is invalid")
    _sha256_text(packet["handoff_plan_sha256"], "handoff plan SHA-256")
    _sha256_text(packet["source_packet_sha256"], "source packet SHA-256")
    _sha256_text(packet["decision_state_sha256"], "decision state SHA-256")
    verify_story_record(packet["rolling_story"])
    recent_events = packet.get("recent_events")
    if not isinstance(recent_events, list):
        raise DojoFreshModelHandoffError("recent events must be an array")
    _validate_recent_events(recent_events)
    if packet.get("cadence_mode") not in {
        "NORMAL_60M",
        "HIGH_RISK_15M",
        "MAJOR_EVENT_IMMEDIATE_ON_COMPILER_INVOCATION",
    }:
        raise DojoFreshModelHandoffError("cadence mode is invalid")
    if len(canonical_json_bytes(packet)) > MAX_PACKET_BYTES:
        raise DojoFreshModelHandoffError("fresh-task decision packet is too large")
    return packet


def _skip_receipt(
    *,
    plan: Mapping[str, Any],
    decision_state_sha256: str,
    reason: str,
) -> dict[str, Any]:
    body = {
        "contract": SKIP_RECEIPT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "handoff_plan_sha256": plan["handoff_plan_sha256"],
        "decision_state_sha256": decision_state_sha256,
        "state": reason,
        "model_tokens_used": 0,
        "fresh_task_created": False,
        "authority": dict(AUTHORITY),
    }
    return {**body, "skip_receipt_sha256": canonical_sha256(body)}


def compile_snapshot(
    *,
    root: Path,
    source_packet: Mapping[str, Any],
    recent_events: Sequence[Mapping[str, Any]] = (),
    risk_signals: Sequence[str] = (),
    major_event_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Publish one fresh-task packet, or return an idempotent zero-token skip."""

    descriptor = _lock(root)
    try:
        plan = _verify_plan(_read_json(root / "handoff-plan.json"))
        events, _ = _verify_events(root, plan)
        status = _status_unlocked(root)
        (
            snapshot,
            source_packet_sha,
            source_recent_events,
            source_cutoff,
        ) = _normalize_source_snapshot(
            source_packet,
            risk_signals=risk_signals,
        )
        bounded_events = _validate_recent_events(
            [*source_recent_events, *recent_events],
            cutoff_utc=source_cutoff,
        )
        majors = sorted(set(major_event_ids))
        if any(event not in MAJOR_EVENT_IDS for event in majors):
            raise DojoFreshModelHandoffError("unknown major event id")
        decision_state = {
            "snapshot": snapshot,
            "recent_events": bounded_events,
            "major_event_ids": majors,
        }
        state_sha = canonical_sha256(decision_state)
        position_count = sum(len(room["positions"]) for room in snapshot["rooms"])
        order_count = sum(len(room["orders"]) for room in snapshot["rooms"])
        market_open = snapshot["market_status"]["is_fx_open"]
        reason: str | None = None
        if status["state"] == "WAITING_FOR_FRESH_TASK":
            if status["current_decision_state_sha256"] == state_sha:
                return status
            raise DojoFreshModelHandoffError(
                "a different state arrived while a fresh task response is pending"
            )
        if not market_open and position_count == 0 and order_count == 0 and not majors:
            reason = "MARKET_CLOSED_FLAT_NO_MODEL_CALL"
        elif position_count == 0 and order_count == 0 and not majors:
            reason = "FLAT_IDLE_NO_MODEL_CALL"
        elif status["last_accepted_decision_state_sha256"] == state_sha and not majors:
            reason = "STATE_HASH_UNCHANGED_NO_MODEL_CALL"
        if reason:
            receipt = _skip_receipt(
                plan=plan,
                decision_state_sha256=state_sha,
                reason=reason,
            )
            _same_or_write(
                root / "skips" / f"{state_sha}-{reason}.json",
                receipt,
            )
            return receipt
        cadence_mode = (
            "MAJOR_EVENT_IMMEDIATE_ON_COMPILER_INVOCATION"
            if majors
            else (
                "HIGH_RISK_15M" if set(risk_signals) & RISK_SIGNAL_IDS else "NORMAL_60M"
            )
        )
        story = _latest_story(root, events)
        body = _packet_body(
            plan=plan,
            snapshot=snapshot,
            source_packet_sha256=source_packet_sha,
            story=story,
            previous_decision=_previous_decision_summary(root, events),
            recent_events=bounded_events,
            cadence_mode=cadence_mode,
            decision_state_sha256=state_sha,
        )
        packet = verify_decision_packet(
            {**body, "decision_packet_sha256": canonical_sha256(body)}
        )
        packet_sha = packet["decision_packet_sha256"]
        _write_exclusive(root / "packets" / f"{packet_sha}.json", packet)
        _same_or_write(root / "ready" / f"{packet_sha}.json", packet)
        _append_event(
            root,
            plan=plan,
            event_type="PACKET_READY",
            payload={
                "decision_packet_sha256": packet_sha,
                "decision_state_sha256": state_sha,
                "cadence_mode": cadence_mode,
            },
        )
        return _status_unlocked(root)
    finally:
        _unlock(descriptor)


def current_ready_packet(root: Path) -> dict[str, Any]:
    descriptor = _lock(root)
    try:
        status = _status_unlocked(root)
        packet_sha = status["current_ready_packet_sha256"]
        if packet_sha is None:
            raise DojoFreshModelHandoffError("no fresh-task packet is ready")
        return verify_decision_packet(_read_json(root / "ready" / f"{packet_sha}.json"))
    finally:
        _unlock(descriptor)


def seal_model_response(
    *,
    packet: Mapping[str, Any],
    action: str,
    reason_ids: Sequence[str],
    next_story_content: Mapping[str, Any],
    provider_model: str,
    provider_execution_id: str,
) -> dict[str, Any]:
    verified = verify_decision_packet(packet)
    reasons = list(reason_ids)
    next_story = validate_story_content(next_story_content)
    if action not in ACTION_IDS:
        raise DojoFreshModelHandoffError("model action is outside allowlist")
    if next_story["last_action"] != action:
        raise DojoFreshModelHandoffError("story last_action must match response")
    if (
        not reasons
        or len(reasons) > 8
        or any(
            not isinstance(reason, str) or not reason or len(reason) > 96
            for reason in reasons
        )
    ):
        raise DojoFreshModelHandoffError("response reason_ids are invalid")
    if (
        not isinstance(provider_model, str)
        or not provider_model
        or len(provider_model) > 128
        or not isinstance(provider_execution_id, str)
        or not provider_execution_id
        or len(provider_execution_id) > 256
    ):
        raise DojoFreshModelHandoffError("provider execution identity is invalid")
    body = {
        "contract": MODEL_RESPONSE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "decision_packet_sha256": verified["decision_packet_sha256"],
        "decision_state_sha256": verified["decision_state_sha256"],
        "current_story_sha256": verified["rolling_story"]["story_sha256"],
        "action": action,
        "reason_ids": reasons,
        "next_story_content": next_story,
        "provider_model": provider_model,
        "provider_execution_id": provider_execution_id,
        "provider_execution_kind": "CODEX_FRESH_TASK",
        "fresh_task_no_conversation_history": True,
        "read_scope_respected": True,
        "future_information_used": False,
        "terminal_outcome_used": False,
        "append_wall_clock_used": False,
        "content_seal_present": True,
        "cryptographic_signature_present": False,
        "model_effect": "SHADOW_RECOMMENDATION_ONLY",
        "classification": "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "authority": dict(AUTHORITY),
    }
    return {**body, "response_sha256": canonical_sha256(body)}


def verify_model_response(
    value: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    response = dict(value)
    verified = verify_decision_packet(packet)
    expected_keys = {
        "contract",
        "schema_version",
        "decision_packet_sha256",
        "decision_state_sha256",
        "current_story_sha256",
        "action",
        "reason_ids",
        "next_story_content",
        "provider_model",
        "provider_execution_id",
        "provider_execution_kind",
        "fresh_task_no_conversation_history",
        "read_scope_respected",
        "future_information_used",
        "terminal_outcome_used",
        "append_wall_clock_used",
        "content_seal_present",
        "cryptographic_signature_present",
        "model_effect",
        "classification",
        "authority",
        "response_sha256",
    }
    unsigned = {key: item for key, item in response.items() if key != "response_sha256"}
    if (
        set(response) != expected_keys
        or response.get("contract") != MODEL_RESPONSE_CONTRACT
        or response.get("schema_version") != SCHEMA_VERSION
        or response.get("decision_packet_sha256") != verified["decision_packet_sha256"]
        or response.get("decision_state_sha256") != verified["decision_state_sha256"]
        or response.get("current_story_sha256")
        != verified["rolling_story"]["story_sha256"]
        or response.get("action") not in ACTION_IDS
        or response.get("provider_execution_kind") != "CODEX_FRESH_TASK"
        or response.get("fresh_task_no_conversation_history") is not True
        or response.get("read_scope_respected") is not True
        or response.get("future_information_used") is not False
        or response.get("terminal_outcome_used") is not False
        or response.get("append_wall_clock_used") is not False
        or response.get("content_seal_present") is not True
        or response.get("cryptographic_signature_present") is not False
        or response.get("model_effect") != "SHADOW_RECOMMENDATION_ONLY"
        or response.get("classification") != "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
        or response.get("authority") != dict(AUTHORITY)
        or response.get("response_sha256") != canonical_sha256(unsigned)
    ):
        raise DojoFreshModelHandoffError("fresh-task response seal is invalid")
    reasons = response.get("reason_ids")
    if (
        not isinstance(reasons, list)
        or not reasons
        or len(reasons) > 8
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise DojoFreshModelHandoffError("response reason ids are invalid")
    story = validate_story_content(response.get("next_story_content"))
    if story["last_action"] != response["action"]:
        raise DojoFreshModelHandoffError("response/story action mismatch")
    return response


def submit_model_response(
    *,
    root: Path,
    response_value: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = _lock(root)
    try:
        status = _status_unlocked(root)
        candidate_sha = response_value.get("decision_packet_sha256")
        if isinstance(candidate_sha, str):
            response_path = root / "responses" / f"{candidate_sha}.json"
            if response_path.exists():
                if _read_json(response_path) != dict(response_value):
                    raise DojoFreshModelHandoffError(
                        "duplicate response conflicts with accepted bytes"
                    )
                return status
        packet_sha = status["current_ready_packet_sha256"]
        if packet_sha is None:
            raise DojoFreshModelHandoffError("no matching fresh-task packet is ready")
        packet = verify_decision_packet(
            _read_json(root / "ready" / f"{packet_sha}.json")
        )
        response = verify_model_response(response_value, packet)
        _write_exclusive(root / "responses" / f"{packet_sha}.json", response)
        plan = _verify_plan(_read_json(root / "handoff-plan.json"))
        events, _ = _verify_events(root, plan)
        prior_story = _latest_story(root, events)
        next_story = _build_story_record(
            story_sequence=prior_story["story_sequence"] + 1,
            previous_story_sha256=prior_story["story_sha256"],
            content=response["next_story_content"],
        )
        _write_exclusive(
            root
            / "stories"
            / (
                f"{next_story['story_sequence']:06d}-"
                f"{next_story['story_sha256']}.json"
            ),
            next_story,
        )
        _append_event(
            root,
            plan=plan,
            event_type="RESPONSE_ACCEPTED",
            payload={
                "decision_packet_sha256": packet_sha,
                "decision_state_sha256": packet["decision_state_sha256"],
                "response_sha256": response["response_sha256"],
                "provider_model": response["provider_model"],
                "provider_execution_id": response["provider_execution_id"],
                "next_story_sha256": next_story["story_sha256"],
                "content_seal_verified": True,
                "cryptographic_signature_verified": False,
            },
        )
        return _status_unlocked(root)
    finally:
        _unlock(descriptor)


def verify_handoff(root: Path) -> dict[str, Any]:
    descriptor = _lock(root)
    try:
        plan = _verify_plan(_read_json(root / "handoff-plan.json"))
        events, _ = _verify_events(root, plan)
        prior_story_sha = ZERO_SHA256
        for sequence, path in enumerate(sorted((root / "stories").glob("*.json"))):
            story = verify_story_record(_read_json(path))
            if (
                story["story_sequence"] != sequence
                or story["previous_story_sha256"] != prior_story_sha
            ):
                raise DojoFreshModelHandoffError("story chain is invalid")
            prior_story_sha = story["story_sha256"]
        for path in sorted((root / "packets").glob("*.json")):
            packet = verify_decision_packet(_read_json(path))
            if path.name != f"{packet['decision_packet_sha256']}.json":
                raise DojoFreshModelHandoffError("packet filename seal mismatch")
        for path in sorted((root / "responses").glob("*.json")):
            response = _read_json(path)
            packet_sha = response.get("decision_packet_sha256")
            packet = _read_json(root / "packets" / f"{packet_sha}.json")
            verify_model_response(response, packet)
        _latest_story(root, events)
        return _status_unlocked(root)
    finally:
        _unlock(descriptor)


__all__ = [
    "DECISION_PACKET_CONTRACT",
    "DojoFreshModelHandoffError",
    "MODEL_RESPONSE_CONTRACT",
    "STORY_CONTRACT",
    "build_paper_source_packet_from_rooms",
    "compile_snapshot",
    "current_ready_packet",
    "handoff_status",
    "initial_story_content",
    "initialize_handoff",
    "seal_model_response",
    "submit_model_response",
    "validate_story_content",
    "verify_decision_packet",
    "verify_handoff",
    "verify_model_response",
    "verify_story_record",
]

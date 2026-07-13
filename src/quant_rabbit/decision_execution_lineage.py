from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.paths import ROOT


DEFAULT_MARKET_READ_EXECUTION_LINKS = ROOT / "data" / "market_read_execution_links.jsonl"

DECISION_EXECUTION_LINEAGE_CONTRACT = "quant_rabbit.market_read_execution_link.v1"
DECISION_EXECUTION_LINEAGE_SCHEMA_VERSION = 1

# Broker client-extension ids are capped independently of the local evidence
# artifacts.  The short token is a content address for the two full 256-bit
# ids; the complete ids remain in the one-shot claim, local request receipt,
# and append-only execution-link artifact.
BROKER_LINEAGE_TOKEN_HEX_LENGTH = 20

# These are evidence-storage engineering bounds, not market thresholds.  They
# keep a malformed response or runaway artifact from consuming the live disk;
# replace them with a managed evidence store if this history outgrows one file.
MAX_EXECUTION_LINK_FILE_BYTES = 64 * 1024 * 1024
MAX_EXECUTION_LINK_LINE_BYTES = 64 * 1024

_GPT_DECISION_ID_RE = re.compile(r"^gptd:[0-9a-f]{64}$")
_MARKET_READ_PREDICTION_ID_RE = re.compile(r"^mr2:[0-9a-f]{64}$")


class DecisionExecutionLineageError(RuntimeError):
    """Exact GPT-to-broker lineage could not be validated or persisted."""


@dataclass(frozen=True)
class DecisionExecutionLineage:
    decision_receipt_id: str
    market_read_prediction_id: str
    lineage_token: str
    decision_generated_at_utc: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_receipt_id": self.decision_receipt_id,
            "market_read_prediction_id": self.market_read_prediction_id,
            "lineage_token": self.lineage_token,
            "decision_generated_at_utc": self.decision_generated_at_utc,
            "identity_basis": "CONTENT_ADDRESSED_VERIFIED_GPT_DECISION_AND_MARKET_READ",
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_utc(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _issue_codes(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    codes: list[str] = []
    for issue in value:
        if not isinstance(issue, Mapping):
            continue
        code = str(issue.get("code") or "").strip()
        if code and code not in codes:
            codes.append(code)
    return codes


def _expected_decision_receipt_id(payload: Mapping[str, Any]) -> str | None:
    decision = payload.get("decision")
    packet = payload.get("input_packet")
    if not isinstance(decision, Mapping) or not isinstance(packet, Mapping):
        return None
    broker = packet.get("broker_snapshot")
    broker = broker if isinstance(broker, Mapping) else {}
    generated_at = _normalize_utc(decision.get("generated_at_utc"))
    receipt_content = {
        "decision_payload": dict(decision),
        "generated_at_utc": generated_at,
        "source_snapshot_at_utc": broker.get("fetched_at_utc"),
        "market_read_first": decision.get("market_read_first", {}),
        "action": str(decision.get("action") or ""),
        "selected_lane_id": decision.get("selected_lane_id"),
        "selected_lane_ids": _string_list(decision.get("selected_lane_ids")),
        "cancel_order_ids": _string_list(decision.get("cancel_order_ids")),
        "close_trade_ids": _string_list(decision.get("close_trade_ids")),
        "confidence": decision.get("confidence"),
        "thesis": decision.get("thesis"),
        "method": decision.get("method"),
        "evidence_refs": _string_list(decision.get("evidence_refs")),
        "verification_status": str(payload.get("status") or ""),
        "verification_issue_codes": _issue_codes(payload.get("verification_issues")),
    }
    return "gptd:" + _sha256(receipt_content)


def decision_lineage_from_verified_payload(
    payload: Mapping[str, Any],
    *,
    selected_lane_id: str,
) -> DecisionExecutionLineage | None:
    """Return exact lineage only for this accepted schema-v2 TRADE lane.

    Non-entry or unrelated verified artifacts deliberately return ``None``;
    callers must not attach an old WAIT/CLOSE decision to a new order.  A
    malformed lineage on an otherwise accepted selected TRADE is an error so
    the live gateway can fail closed instead of silently losing attribution.
    """

    if str(payload.get("status") or "").strip().upper() != "ACCEPTED":
        return None
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        return None
    action = str(decision.get("action") or "").strip().upper()
    if action != "TRADE":
        return None
    selected = _string_list(decision.get("selected_lane_ids"))
    primary = str(decision.get("selected_lane_id") or "").strip()
    if primary:
        selected.append(primary)
    selected = list(dict.fromkeys(selected))
    if not selected_lane_id or selected_lane_id not in selected:
        return None

    provenance = decision.get("decision_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    codex_market_read = (
        str(provenance.get("author_kind") or "").strip().upper()
        == "CODEX_MARKET_READ"
    )
    recorded = payload.get("market_read_prediction")
    if not isinstance(recorded, Mapping):
        # Historical/unit fixtures may carry an accepted deterministic receipt
        # that predates the content-addressed market-read author contract.  It
        # gets no lineage claim.  A current CODEX_MARKET_READ receipt, however,
        # must have been recorded by GPTTraderBrain and fails closed if absent.
        if not codex_market_read:
            return None
        raise DecisionExecutionLineageError(
            "accepted CODEX_MARKET_READ entry is missing market_read_prediction lineage"
        )
    decision_receipt_id = str(recorded.get("decision_receipt_id") or "").strip()
    prediction_id = str(
        recorded.get("market_read_prediction_id")
        or recorded.get("prediction_id")
        or ""
    ).strip()
    if not _GPT_DECISION_ID_RE.fullmatch(decision_receipt_id):
        raise DecisionExecutionLineageError(
            "accepted GPT entry has an invalid decision_receipt_id"
        )
    if not _MARKET_READ_PREDICTION_ID_RE.fullmatch(prediction_id):
        raise DecisionExecutionLineageError(
            "accepted GPT entry has an invalid market_read_prediction_id"
        )
    if recorded.get("live_permission") is not False:
        raise DecisionExecutionLineageError(
            "market-read measurement must remain read-only and cannot grant live permission"
        )
    expected_receipt_id = _expected_decision_receipt_id(payload)
    if expected_receipt_id is None or expected_receipt_id != decision_receipt_id:
        raise DecisionExecutionLineageError(
            "decision_receipt_id no longer matches the verified decision content"
        )
    token = "mdl-" + hashlib.sha256(
        f"{decision_receipt_id}\0{prediction_id}".encode("utf-8")
    ).hexdigest()[:BROKER_LINEAGE_TOKEN_HEX_LENGTH]
    return DecisionExecutionLineage(
        decision_receipt_id=decision_receipt_id,
        market_read_prediction_id=prediction_id,
        lineage_token=token,
        decision_generated_at_utc=_normalize_utc(decision.get("generated_at_utc")),
    )


def read_verified_decision_lineage(
    path: Path | None,
    *,
    selected_lane_id: str,
) -> DecisionExecutionLineage | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionExecutionLineageError(
            f"verified GPT decision lineage is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DecisionExecutionLineageError("verified GPT decision is not a JSON object")
    return decision_lineage_from_verified_payload(
        payload,
        selected_lane_id=selected_lane_id,
    )


def attach_lineage_metadata(
    metadata: Mapping[str, Any] | None,
    lineage: DecisionExecutionLineage | None,
) -> dict[str, Any]:
    out = dict(metadata or {})
    if lineage is None:
        return out
    for key, value in (
        ("gpt_decision_receipt_id", lineage.decision_receipt_id),
        ("market_read_prediction_id", lineage.market_read_prediction_id),
        ("gpt_decision_lineage_token", lineage.lineage_token),
    ):
        existing = str(out.get(key) or "").strip()
        if existing and existing != value:
            raise DecisionExecutionLineageError(
                f"intent metadata {key} conflicts with the current verified GPT decision"
            )
        out[key] = value
    return out


def lineage_from_metadata(metadata: Mapping[str, Any] | None) -> DecisionExecutionLineage | None:
    metadata = metadata or {}
    decision_receipt_id = str(metadata.get("gpt_decision_receipt_id") or "").strip()
    prediction_id = str(metadata.get("market_read_prediction_id") or "").strip()
    token = str(metadata.get("gpt_decision_lineage_token") or "").strip()
    if not decision_receipt_id and not prediction_id and not token:
        return None
    if not _GPT_DECISION_ID_RE.fullmatch(decision_receipt_id):
        raise DecisionExecutionLineageError("intent has invalid gpt_decision_receipt_id")
    if not _MARKET_READ_PREDICTION_ID_RE.fullmatch(prediction_id):
        raise DecisionExecutionLineageError("intent has invalid market_read_prediction_id")
    expected_token = "mdl-" + hashlib.sha256(
        f"{decision_receipt_id}\0{prediction_id}".encode("utf-8")
    ).hexdigest()[:BROKER_LINEAGE_TOKEN_HEX_LENGTH]
    if token != expected_token:
        raise DecisionExecutionLineageError("intent decision-lineage token does not match its full ids")
    return DecisionExecutionLineage(
        decision_receipt_id=decision_receipt_id,
        market_read_prediction_id=prediction_id,
        lineage_token=token,
        decision_generated_at_utc=None,
    )


def broker_identifiers_from_gateway_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only broker-explicit ids from the actual order POST response."""

    order_ids: list[str] = []
    fill_transaction_ids: list[str] = []
    trade_ids: list[str] = []
    transaction_ids: list[str] = []

    def add(target: list[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)

    for key in (
        "orderCreateTransaction",
        "orderFillTransaction",
        "orderCancelTransaction",
        "orderRejectTransaction",
    ):
        transaction = response.get(key)
        if isinstance(transaction, Mapping):
            add(transaction_ids, transaction.get("id"))

    created = response.get("orderCreateTransaction")
    if isinstance(created, Mapping):
        add(order_ids, created.get("id"))

    fill = response.get("orderFillTransaction")
    if isinstance(fill, Mapping):
        add(fill_transaction_ids, fill.get("id"))
        add(order_ids, fill.get("orderID"))
        opened = fill.get("tradeOpened")
        if isinstance(opened, Mapping):
            add(trade_ids, opened.get("tradeID"))
        reduced = fill.get("tradeReduced")
        if isinstance(reduced, Mapping):
            add(trade_ids, reduced.get("tradeID"))
        closed = fill.get("tradesClosed")
        if isinstance(closed, Sequence) and not isinstance(closed, (str, bytes)):
            for item in closed:
                if isinstance(item, Mapping):
                    add(trade_ids, item.get("tradeID"))

    for value in response.get("relatedTransactionIDs", []) or []:
        add(transaction_ids, value)

    last_transaction_id = str(response.get("lastTransactionID") or "").strip() or None
    return {
        "order_ids": order_ids,
        "fill_transaction_ids": fill_transaction_ids,
        "trade_ids": trade_ids,
        "transaction_ids": transaction_ids,
        "last_transaction_id": last_transaction_id,
    }


def build_execution_link(
    *,
    lineage: DecisionExecutionLineage,
    gateway_response: Mapping[str, Any],
    lane_id: str,
    parent_lane_id: str | None,
    forecast_cycle_id: str | None,
    claim_id: str | None,
    order_request_sha256: str | None,
    client_extension_id: str | None,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    broker_ids = broker_identifiers_from_gateway_response(gateway_response)
    has_explicit_id = any(
        broker_ids.get(key)
        for key in (
            "order_ids",
            "fill_transaction_ids",
            "trade_ids",
            "transaction_ids",
            "last_transaction_id",
        )
    )
    if not has_explicit_id:
        raise DecisionExecutionLineageError(
            "actual gateway response contains no explicit broker order/fill/trade/transaction id"
        )
    immutable = {
        "schema_version": DECISION_EXECUTION_LINEAGE_SCHEMA_VERSION,
        "contract": DECISION_EXECUTION_LINEAGE_CONTRACT,
        **lineage.to_dict(),
        "lane_id": str(lane_id or ""),
        "parent_lane_id": str(parent_lane_id or "") or None,
        "forecast_cycle_id": str(forecast_cycle_id or "") or None,
        "ordinary_entry_claim_id": str(claim_id or "") or None,
        "order_request_sha256": str(order_request_sha256 or "") or None,
        "client_extension_id": str(client_extension_id or "") or None,
        "broker_ids": broker_ids,
        "gateway_response_sha256": _sha256(dict(gateway_response)),
        "attribution_basis": "EXPLICIT_ACTUAL_GATEWAY_RESPONSE_IDS_ONLY",
        "pair_or_time_inference_used": False,
        "live_permission": False,
    }
    if broker_ids["trade_ids"]:
        execution_status = "TRADE_ID_EXPLICIT"
    elif broker_ids["fill_transaction_ids"]:
        execution_status = "FILL_ID_EXPLICIT_TRADE_ID_PENDING"
    else:
        execution_status = "ORDER_ID_EXPLICIT_FILL_PENDING"
    immutable["execution_status"] = execution_status
    return {
        **immutable,
        "link_id": "mrel:" + _sha256(immutable),
        "recorded_at_utc": recorded_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _immutable_link_payload(link: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in link.items()
        if key not in {"link_id", "recorded_at_utc"}
    }


def append_execution_link(path: Path, link: Mapping[str, Any]) -> dict[str, Any]:
    expected_id = "mrel:" + _sha256(_immutable_link_payload(link))
    if str(link.get("link_id") or "") != expected_id:
        raise DecisionExecutionLineageError("execution link_id does not match immutable link content")
    line = _canonical_json(dict(link)) + "\n"
    if len(line.encode("utf-8")) > MAX_EXECUTION_LINK_LINE_BYTES:
        raise DecisionExecutionLineageError("execution link exceeds the bounded JSONL row size")

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            current_size = path.stat().st_size if path.exists() else 0
            if current_size + len(line.encode("utf-8")) > MAX_EXECUTION_LINK_FILE_BYTES:
                raise DecisionExecutionLineageError(
                    "execution-link artifact exceeded its bounded file size"
                )
            existing = read_execution_links(path)
            _ensure_unique_broker_id_ownership([*existing, dict(link)])
            for row in existing:
                if row.get("link_id") != expected_id:
                    continue
                if _immutable_link_payload(row) != _immutable_link_payload(link):
                    raise DecisionExecutionLineageError(
                        "execution-link id collision or mutation detected"
                    )
                return {
                    "status": "COALESCED_EXACT_DUPLICATE",
                    "path": str(path),
                    "link_id": expected_id,
                    "link": row,
                }
            file_existed = path.exists()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            if not file_existed:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return {
                "status": "RECORDED",
                "path": str(path),
                "link_id": expected_id,
                "link": dict(link),
            }
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def read_execution_links(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.stat().st_size > MAX_EXECUTION_LINK_FILE_BYTES:
        raise DecisionExecutionLineageError("execution-link artifact exceeds its bounded file size")
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise DecisionExecutionLineageError(f"execution-link artifact is unreadable: {exc}") from exc
    rows: list[dict[str, Any]] = []
    seen_ids: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        if len(raw) > MAX_EXECUTION_LINK_LINE_BYTES:
            raise DecisionExecutionLineageError(
                f"execution-link row {index} exceeds the bounded row size"
            )
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DecisionExecutionLineageError(
                f"execution-link artifact is malformed at row {index}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} is not an object"
            )
        if row.get("schema_version") != DECISION_EXECUTION_LINEAGE_SCHEMA_VERSION:
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has unsupported schema_version"
            )
        if row.get("contract") != DECISION_EXECUTION_LINEAGE_CONTRACT:
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has an invalid contract"
            )
        decision_receipt_id = str(row.get("decision_receipt_id") or "")
        prediction_id = str(row.get("market_read_prediction_id") or "")
        if not _GPT_DECISION_ID_RE.fullmatch(decision_receipt_id):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has invalid decision_receipt_id"
            )
        if not _MARKET_READ_PREDICTION_ID_RE.fullmatch(prediction_id):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has invalid market_read_prediction_id"
            )
        expected_token = "mdl-" + hashlib.sha256(
            f"{decision_receipt_id}\0{prediction_id}".encode("utf-8")
        ).hexdigest()[:BROKER_LINEAGE_TOKEN_HEX_LENGTH]
        if row.get("lineage_token") != expected_token:
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has invalid lineage_token"
            )
        if row.get("pair_or_time_inference_used") is not False:
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} does not prohibit pair/time inference"
            )
        if row.get("attribution_basis") != "EXPLICIT_ACTUAL_GATEWAY_RESPONSE_IDS_ONLY":
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has an invalid attribution basis"
            )
        if row.get("live_permission") is not False:
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} incorrectly claims live permission"
            )
        broker_ids = row.get("broker_ids")
        if not isinstance(broker_ids, Mapping):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} lacks explicit broker ids"
            )
        list_fields = (
            "order_ids",
            "fill_transaction_ids",
            "trade_ids",
            "transaction_ids",
        )
        if any(not isinstance(broker_ids.get(field), list) for field in list_fields):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has malformed broker id lists"
            )
        if not any(broker_ids.get(field) for field in list_fields) and not str(
            broker_ids.get("last_transaction_id") or ""
        ):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} contains no explicit broker id"
            )
        expected_id = "mrel:" + _sha256(_immutable_link_payload(row))
        if row.get("link_id") != expected_id:
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} failed its content digest"
            )
        prior = seen_ids.get(expected_id)
        if prior is not None and _immutable_link_payload(prior) != _immutable_link_payload(row):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} conflicts with an earlier link id"
            )
        seen_ids[expected_id] = row
        rows.append(row)
    _ensure_unique_broker_id_ownership(rows)
    return rows


def _ensure_unique_broker_id_ownership(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject one broker identity claimed by different GPT prediction owners.

    OANDA order, fill, trade, and transaction ids are explicit broker
    identities.  A later append may legitimately repeat one of those ids while
    enriching the *same* ``(gptd, mr2)`` owner, but assigning it to another
    owner would make realized P/L ambiguous and must invalidate the artifact.
    """

    owners_by_broker_id: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows, start=1):
        decision_receipt_id = str(row.get("decision_receipt_id") or "").strip()
        prediction_id = str(row.get("market_read_prediction_id") or "").strip()
        if not _GPT_DECISION_ID_RE.fullmatch(decision_receipt_id) or not (
            _MARKET_READ_PREDICTION_ID_RE.fullmatch(prediction_id)
        ):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} has malformed lineage ownership"
            )
        owner = (decision_receipt_id, prediction_id)
        broker_ids = row.get("broker_ids")
        if not isinstance(broker_ids, Mapping):
            raise DecisionExecutionLineageError(
                f"execution-link artifact row {index} lacks explicit broker ids"
            )

        explicit_ids: list[str] = []
        for field in (
            "order_ids",
            "fill_transaction_ids",
            "trade_ids",
            "transaction_ids",
        ):
            values = broker_ids.get(field)
            if not isinstance(values, list):
                raise DecisionExecutionLineageError(
                    f"execution-link artifact row {index} has malformed broker id lists"
                )
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise DecisionExecutionLineageError(
                        f"execution-link artifact row {index} has malformed broker id values"
                    )
                explicit_ids.append(value.strip())
        last_transaction_id = broker_ids.get("last_transaction_id")
        if last_transaction_id is not None:
            if not isinstance(last_transaction_id, str) or not last_transaction_id.strip():
                raise DecisionExecutionLineageError(
                    f"execution-link artifact row {index} has malformed last_transaction_id"
                )
            # This is an account progress watermark, not exclusive order/fill/
            # trade ownership. Distinct responses may legitimately observe the
            # same watermark, so validate its shape without reserving it.

        for broker_id in dict.fromkeys(explicit_ids):
            prior_owner = owners_by_broker_id.get(broker_id)
            if prior_owner is not None and prior_owner != owner:
                raise DecisionExecutionLineageError(
                    "execution-link artifact has conflicting lineage owners for "
                    f"broker id {broker_id}: {prior_owner[0]}/{prior_owner[1]} vs "
                    f"{owner[0]}/{owner[1]}"
                )
            owners_by_broker_id[broker_id] = owner

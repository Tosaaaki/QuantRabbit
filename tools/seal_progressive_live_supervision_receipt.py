#!/usr/bin/env python3
"""Seal one order-field-free LLM supervision receipt for an exact preflight event.

The caller (the owner heartbeat model) may choose only regime, allowed strategy
ids, a risk-budget cap, a position cap, FREEZE/UNWIND, and expiry.  Pair, side,
units, price, TP/SL, and order type are taken from neither CLI arguments nor the
receipt.  They remain sealed fast-bot signal and deterministic sizing fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot import SIGNAL_CONTRACT  # noqa: E402
from quant_rabbit.fast_bot_promotion import (  # noqa: E402
    SUPERVISION_RECEIPT_CONTRACT,
    seal_supervision_receipt,
)

_PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "qr_progressive_live_preflight_for_supervision",
    ROOT / "tools" / "run_progressive_live_preflight.py",
)
if _PREFLIGHT_SPEC is None or _PREFLIGHT_SPEC.loader is None:
    raise RuntimeError("PREFLIGHT_IMPORT_FAILED")
_PREFLIGHT = importlib.util.module_from_spec(_PREFLIGHT_SPEC)
_PREFLIGHT_SPEC.loader.exec_module(_PREFLIGHT)

_SEALER_SPEC = importlib.util.spec_from_file_location(
    "qr_progressive_live_sealer_for_supervision",
    ROOT / "tools" / "seal_progressive_live_risk_contract.py",
)
if _SEALER_SPEC is None or _SEALER_SPEC.loader is None:
    raise RuntimeError("RELEASE_SEALER_IMPORT_FAILED")
_RELEASE_SEALER = importlib.util.module_from_spec(_SEALER_SPEC)
_SEALER_SPEC.loader.exec_module(_RELEASE_SEALER)

MAX_SUPERVISION_SECONDS = 6 * 60 * 60
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{0,63}$")


class SupervisionSealBlocked(RuntimeError):
    """The AI receipt cannot be bound to current exact promotion evidence."""


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_utc(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SupervisionSealBlocked("TIME_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SupervisionSealBlocked("TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc)


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SupervisionSealBlocked(f"{name}_INVALID") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise SupervisionSealBlocked(f"{name}_INVALID")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise SupervisionSealBlocked(f"{name}_INVALID")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SupervisionSealBlocked(f"{name}_INVALID") from exc
    if parsed <= 0:
        raise SupervisionSealBlocked(f"{name}_INVALID")
    return parsed


def _sealed(value: Mapping[str, Any], *, contract: str, seal_key: str) -> bool:
    if value.get("contract") != contract:
        return False
    body = {key: item for key, item in value.items() if key != seal_key}
    return value.get(seal_key) == _canonical_sha(body)


def _event_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("contract") != "QR_PROGRESSIVE_LIVE_PREFLIGHT_V1":
        raise SupervisionSealBlocked("PREFLIGHT_STATE_CONTRACT_INVALID")
    event_id = str(state.get("last_event_id") or "")
    ledger = Path(str(state.get("ledger_path") or ""))
    if not event_id or not ledger.is_file():
        raise SupervisionSealBlocked("PREFLIGHT_MODE_EVENT_MISSING")
    for line in reversed(ledger.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SupervisionSealBlocked("PREFLIGHT_LEDGER_INVALID") from exc
        if isinstance(row, dict) and row.get("event_id") == event_id:
            body = {key: item for key, item in row.items() if key != "event_id"}
            if event_id != f"qrplm:{_canonical_sha(body)}":
                raise SupervisionSealBlocked("PREFLIGHT_EVENT_SEAL_INVALID")
            return row
    raise SupervisionSealBlocked("PREFLIGHT_MODE_EVENT_NOT_FOUND")


def _selected_signal(event: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in event.get("signal_receipts") or []
        if isinstance(row, Mapping)
        and isinstance(row.get("signal"), Mapping)
        and (row.get("mode_receipt") or {}).get("mode") == "THROTTLED_LIVE"
        and int((row.get("mode_receipt") or {}).get("calculated_units") or 0) > 0
    ]
    if not rows:
        raise SupervisionSealBlocked("THROTTLED_SIGNAL_RECEIPT_MISSING")
    selected = max(
        rows,
        key=lambda row: int((row.get("mode_receipt") or {}).get("calculated_units") or 0),
    )
    signal = dict(selected["signal"])
    if not _sealed(signal, contract=SIGNAL_CONTRACT, seal_key="signal_sha256"):
        raise SupervisionSealBlocked("SIGNAL_SEAL_INVALID")
    return signal


def build_progressive_live_supervision_receipt(
    *,
    preflight_state: Mapping[str, Any],
    event: Mapping[str, Any],
    release_receipt: Mapping[str, Any],
    software_manifest: Mapping[str, Any],
    expected_packet_sha256: str,
    decision: str,
    regime: str,
    allowed_strategy_ids: Sequence[str],
    risk_budget_cap_jpy: float,
    max_positions_cap: int,
    expiry_seconds: int,
    review_reason: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """Bind a bounded LLM decision to one exact GET-only preflight event."""

    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise SupervisionSealBlocked("NOW_UTC_TIMEZONE_REQUIRED")
    now = now_utc.astimezone(timezone.utc)
    event_id = str(event.get("event_id") or "")
    if (
        preflight_state.get("last_event_id") != event_id
        or event.get("mode") != preflight_state.get("mode")
        or event.get("software_version_sha256")
        != software_manifest.get("software_version_sha256")
    ):
        raise SupervisionSealBlocked("PREFLIGHT_EVENT_BINDING_INVALID")
    admission, risk = _PREFLIGHT.verify_release_receipt(
        release_receipt,
        approval_packet_sha256=expected_packet_sha256,
        software_manifest=software_manifest,
    )
    if event.get("release_receipt_sha256") != release_receipt.get("release_receipt_sha256"):
        raise SupervisionSealBlocked("PREFLIGHT_RELEASE_BINDING_INVALID")

    normalized_decision = str(decision or "").upper()
    normalized_regime = str(regime or "").upper()
    reason = str(review_reason or "").strip()
    if normalized_decision not in {"ALLOW", "FREEZE_NEW", "UNWIND"}:
        raise SupervisionSealBlocked("DECISION_INVALID")
    if _IDENTIFIER_RE.fullmatch(normalized_regime) is None:
        raise SupervisionSealBlocked("REGIME_INVALID")
    if not 1 <= len(reason) <= 500:
        raise SupervisionSealBlocked("REVIEW_REASON_INVALID")
    expiry = _positive_int(expiry_seconds, "EXPIRY_SECONDS")
    if expiry > MAX_SUPERVISION_SECONDS:
        raise SupervisionSealBlocked("EXPIRY_SECONDS_EXCEEDS_SIX_HOURS")

    strategies = list(allowed_strategy_ids)
    risk_cap = float(risk_budget_cap_jpy)
    position_cap = int(max_positions_cap)
    signal: dict[str, Any] | None = None
    if normalized_decision == "ALLOW":
        if preflight_state.get("promotion_ready") is not True or event.get("mode") != "THROTTLED_LIVE":
            raise SupervisionSealBlocked("ALLOW_REQUIRES_CURRENT_THROTTLED_PREFLIGHT")
        signal = _selected_signal(event)
        source_strategy = str(signal.get("strategy_id") or "")
        selected_live_strategy = f"live-{source_strategy}"
        if (
            not strategies
            or len(strategies) != len(set(strategies))
            or selected_live_strategy not in strategies
            or any(
                not item.startswith("live-")
                or item.removeprefix("live-") not in admission.get("allowed_strategy_ids", [])
                for item in strategies
            )
        ):
            raise SupervisionSealBlocked("ALLOWED_STRATEGY_IDS_INVALID")
        risk_cap = _positive_float(risk_cap, "RISK_BUDGET_CAP_JPY")
        position_cap = _positive_int(position_cap, "MAX_POSITIONS_CAP")
        if risk_cap > float(risk["max_loss_per_order_jpy"]):
            raise SupervisionSealBlocked("RISK_BUDGET_EXCEEDS_USER_CONTRACT")
        if position_cap > int(risk["max_bot_positions"]):
            raise SupervisionSealBlocked("POSITION_CAP_EXCEEDS_USER_CONTRACT")
    else:
        if strategies or risk_cap != 0.0 or position_cap != 0:
            raise SupervisionSealBlocked("FREEZE_OR_UNWIND_MUST_HAVE_ZERO_ENTRY_CAPACITY")
        signal = _selected_signal(event) if event.get("mode") == "THROTTLED_LIVE" else None

    generated_at = now.isoformat()
    expires_at = now + timedelta(seconds=expiry)
    signal_sha = str((signal or {}).get("signal_sha256") or "")
    regime_sha = str((signal or {}).get("regime_contract_sha256") or "")
    if normalized_decision == "ALLOW":
        quote_at = _parse_utc(signal.get("quote_timestamp_utc"))
        signal_expiry = quote_at + timedelta(
            seconds=_positive_int(signal.get("entry_ttl_seconds"), "SIGNAL_TTL_SECONDS")
        )
        expires_at = min(expires_at, signal_expiry)
        if expires_at <= now or not _SHA256_RE.fullmatch(regime_sha):
            raise SupervisionSealBlocked("SIGNAL_EXPIRED_OR_REGIME_BINDING_INVALID")

    feature_sha = event_id.removeprefix("qrplm:")
    if not _SHA256_RE.fullmatch(feature_sha):
        raise SupervisionSealBlocked("FEATURE_SNAPSHOT_BINDING_INVALID")
    body = {
        "contract": SUPERVISION_RECEIPT_CONTRACT,
        "receipt_id": "pending",
        "event_id": event_id,
        "dedupe_key": event_id,
        "feature_snapshot_sha256": feature_sha,
        "signal_sha256": signal_sha,
        "regime_contract_sha256": regime_sha,
        "decision": normalized_decision,
        "regime": normalized_regime,
        "allowed_strategy_ids": strategies,
        "risk_budget_cap_jpy": risk_cap,
        "max_positions_cap": position_cap,
        "generated_at_utc": generated_at,
        "expires_at_utc": expires_at.isoformat(),
        "review_reason": reason,
        "ai_role": "REGIME_RISK_AND_INVENTORY_SUPERVISOR_ONLY",
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    receipt_id = f"qrsup:{_canonical_sha({key: value for key, value in body.items() if key != 'receipt_id'})}"
    return seal_supervision_receipt({**body, "receipt_id": receipt_id})


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temp, 0o600)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-state", type=Path, required=True)
    parser.add_argument("--approval-packet", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--decision", choices=("ALLOW", "FREEZE_NEW", "UNWIND"), required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--allowed-strategy-id", action="append", default=[])
    parser.add_argument("--risk-budget-cap-jpy", type=float, default=0.0)
    parser.add_argument("--max-positions-cap", type=int, default=0)
    parser.add_argument("--expiry-seconds", type=int, required=True)
    parser.add_argument("--review-reason", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--now-utc")
    args = parser.parse_args()

    packet = _RELEASE_SEALER.verify_approval_packet(
        _RELEASE_SEALER.read_json(args.approval_packet),
        expected_packet_sha256=args.expected_packet_sha256,
    )
    if packet.get("acceptance", {}).get("accepted_by_user") is not False:
        raise SupervisionSealBlocked("APPROVAL_PACKET_CANDIDATE_STATE_INVALID")
    state = _PREFLIGHT.read_json(args.preflight_state)
    event = _event_from_state(state)
    release = _PREFLIGHT.read_json(args.release_receipt)
    manifest = _RELEASE_SEALER.software_manifest()
    now = _parse_utc(args.now_utc) if args.now_utc else datetime.now(timezone.utc)
    receipt = build_progressive_live_supervision_receipt(
        preflight_state=state,
        event=event,
        release_receipt=release,
        software_manifest=manifest,
        expected_packet_sha256=args.expected_packet_sha256,
        decision=args.decision,
        regime=args.regime,
        allowed_strategy_ids=args.allowed_strategy_id,
        risk_budget_cap_jpy=args.risk_budget_cap_jpy,
        max_positions_cap=args.max_positions_cap,
        expiry_seconds=args.expiry_seconds,
        review_reason=args.review_reason,
        now_utc=now,
    )
    _atomic_json(args.output, receipt)
    print(
        json.dumps(
            {
                "status": "SEALED_PROGRESSIVE_LIVE_SUPERVISION",
                "decision": receipt["decision"],
                "receipt_id": receipt["receipt_id"],
                "event_id": receipt["event_id"],
                "expires_at_utc": receipt["expires_at_utc"],
                "ai_order_authority": receipt["ai_order_authority"],
                "broker_mutation_allowed": receipt["broker_mutation_allowed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

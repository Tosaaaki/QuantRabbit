"""Seal bounded AI regime supervision without granting order authority."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.fast_bot import (
    ACTIVE_AI_MODES,
    AI_SUPERVISION_CONTRACT,
    REGIME_CONTRACT,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


CANDIDATE_CONTRACT = "QR_AI_REGIME_SUPERVISION_CANDIDATE_V1"
MAX_SUPERVISION_SECONDS = 6 * 60 * 60
FORBIDDEN_ORDER_KEYS = {
    "action",
    "side",
    "method",
    "entry",
    "take_profit",
    "stop_loss",
    "units",
    "risk",
    "risk_pct",
    "live_permission",
    "broker_mutation",
    "order",
    "orders",
    "trade",
    "trades",
    "cancel",
    "close",
}
_CANDIDATE_KEYS = {
    "contract",
    "schema_version",
    "reviewed_at_utc",
    "review_reason",
    "regime_contract_sha256",
    "scorecard_contract_sha256",
    "pairs",
}
_PAIR_KEYS = {"mode", "reason", "expires_at_utc"}


def build_ai_regime_supervision(
    candidate: Mapping[str, Any],
    *,
    regime_contract: Mapping[str, Any],
    scorecard: Mapping[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Validate one AI review and emit only pair GO/CAUTION/STOP supervision."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    if set(candidate) != _CANDIDATE_KEYS or _contains_forbidden_key(candidate):
        raise ValueError("candidate contains unsupported or order-authority fields")
    schema_version = candidate.get("schema_version")
    if (
        candidate.get("contract") != CANDIDATE_CONTRACT
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("candidate contract is invalid")
    reviewed = _parse_utc(candidate.get("reviewed_at_utc"))
    if reviewed is None or not -5.0 <= (now - reviewed).total_seconds() <= 60.0:
        raise ValueError("candidate review clock is stale or future-dated")
    review_reason = str(candidate.get("review_reason") or "").strip()
    if not 1 <= len(review_reason) <= 500:
        raise ValueError("candidate review_reason is required and bounded")

    regime_sha = _validated_contract_sha(regime_contract, REGIME_CONTRACT)
    scorecard_sha = _validated_contract_sha(
        scorecard,
        "QR_FAST_BOT_FORWARD_SCORECARD_V1",
    )
    if candidate.get("regime_contract_sha256") != regime_sha:
        raise ValueError("candidate regime contract binding is stale")
    if candidate.get("scorecard_contract_sha256") != scorecard_sha:
        raise ValueError("candidate scorecard binding is stale")

    available_pairs = {
        str(row.get("pair") or "")
        for row in regime_contract.get("rows", [])
        if isinstance(row, Mapping)
    }
    pairs = candidate.get("pairs")
    if not isinstance(pairs, Mapping):
        raise ValueError("candidate pairs must be an object")
    normalized_pairs: dict[str, dict[str, Any]] = {}
    allowed_pairs = set(DEFAULT_TRADER_PAIRS)
    for pair, raw in sorted(pairs.items(), key=lambda item: str(item[0])):
        if not isinstance(pair, str):
            raise ValueError("supervised pair keys must be strings")
        pair_name = pair
        if pair_name not in allowed_pairs or pair_name not in available_pairs:
            raise ValueError(f"unsupported or absent supervised pair: {pair_name}")
        if not isinstance(raw, Mapping) or set(raw) != _PAIR_KEYS:
            raise ValueError(f"{pair_name}: supervision row shape is invalid")
        if _contains_forbidden_key(raw):
            raise ValueError(f"{pair_name}: order-authority field is forbidden")
        mode = str(raw.get("mode") or "").upper()
        reason = str(raw.get("reason") or "").strip()
        expires = _parse_utc(raw.get("expires_at_utc"))
        if mode not in ACTIVE_AI_MODES:
            raise ValueError(f"{pair_name}: mode must be GO, CAUTION, or STOP")
        if not 1 <= len(reason) <= 500:
            raise ValueError(f"{pair_name}: reason is required and bounded")
        if expires is None or not reviewed < expires <= reviewed + timedelta(
            seconds=MAX_SUPERVISION_SECONDS
        ):
            raise ValueError(f"{pair_name}: expiry must be within six hours")
        normalized_pairs[pair_name] = {
            "mode": mode,
            "reason": reason,
            "expires_at_utc": expires.isoformat(),
        }

    body = {
        "contract": AI_SUPERVISION_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "last_tuned_at_utc": reviewed.isoformat(),
        "review_reason": review_reason,
        "regime_contract_sha256": regime_sha,
        "scorecard_contract_sha256": scorecard_sha,
        "ai_role": "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY",
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "pairs": normalized_pairs,
    }
    return _seal(body)


def write_ai_regime_supervision(
    candidate_path: Path,
    regime_path: Path,
    scorecard_path: Path,
    output_path: Path,
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    candidate = _read_object(candidate_path)
    regime = _read_object(regime_path)
    scorecard = _read_object(scorecard_path)
    supervision = build_ai_regime_supervision(
        candidate,
        regime_contract=regime,
        scorecard=scorecard,
        now_utc=now_utc,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(supervision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return supervision


def _validated_contract_sha(value: Mapping[str, Any], contract: str) -> str:
    if value.get("contract") != contract:
        raise ValueError(f"missing current {contract} artifact")
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    if not _sha256_text(stored) or stored != _canonical_sha(body):
        raise ValueError(f"invalid sealed {contract} artifact")
    return stored


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_ORDER_KEYS:
                return True
            if _contains_forbidden_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "CANDIDATE_CONTRACT",
    "build_ai_regime_supervision",
    "write_ai_regime_supervision",
]

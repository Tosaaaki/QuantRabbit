#!/usr/bin/env python3
"""Seal one fixed-schema DOJO response without mounting or reading an answer key."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.dojo_ai_discretion import canonical_sha256, seal_response
from quant_rabbit.dojo_prompt_phase import build_cell_assignment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-json", type=Path, required=True)
    parser.add_argument("--cell-assignment", type=Path, required=True)
    parser.add_argument("--request-receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--prompt-lock", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--sealed-at-utc")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    response_value = _read_object(args.response_json)
    assignment = _read_object(args.cell_assignment)
    request = _read_object(args.request_receipt)
    packet = _read_object(args.packet)
    prompt = _read_object(args.prompt_lock)
    model = _read_object(args.model_manifest)
    capability = _read_object(args.capability_manifest)
    _validate_request_receipt(request)
    rebuilt = build_cell_assignment(
        phase_id=assignment["phase_id"],
        blind_day_rank=assignment["blind_day_rank"],
        blind_day_id=assignment["blind_day_id"],
        variant_id=assignment["variant_id"],
        source_sha256=assignment["source_sha256"],
        packet_sha256=assignment["packet_sha256"],
        prompt_sha256=assignment["prompt_sha256"],
        prompt_lock_sha256=assignment["prompt_lock_sha256"],
        model_sha256=assignment["model_sha256"],
        capability_manifest_sha256=assignment["capability_manifest_sha256"],
        request_receipt_sha256=assignment["request_receipt_sha256"],
        context_id=assignment["context_id"],
    )
    if assignment != rebuilt:
        raise ValueError("cell assignment seal is stale")
    bindings = {
        "cell_id": request.get("cell_id"),
        "packet_sha256": packet.get("packet_sha256"),
        "prompt_sha256": prompt.get("prompt_sha256"),
        "prompt_lock_sha256": prompt.get("prompt_lock_sha256"),
        "model_sha256": model.get("model_sha256"),
        "capability_manifest_sha256": capability.get("capability_manifest_sha256"),
        "request_receipt_sha256": request.get("request_receipt_sha256"),
    }
    for field, actual in bindings.items():
        if assignment.get(field) != actual:
            raise ValueError(f"{field} does not match the allocated cell")

    sealed_at = _parse_or_now(args.sealed_at_utc)
    response = seal_response(
        response_value,
        packet=packet,
        prompt_lock=prompt,
        model_manifest=model,
        capability_manifest=capability,
        sealed_at_utc=sealed_at,
    )
    cell_body = {
        "contract": "QR_DOJO_PROMPT_CELL_RESPONSE_SEAL_V1",
        "schema_version": 1,
        "validity_status": response["validity_status"],
        "cell_id": assignment["cell_id"],
        "phase_id": assignment["phase_id"],
        "blind_day_rank": assignment["blind_day_rank"],
        "blind_day_id": assignment["blind_day_id"],
        "variant_id": assignment["variant_id"],
        "request_receipt_sha256": assignment["request_receipt_sha256"],
        "response_receipt_sha256": response["response_receipt_sha256"],
        "answer_key_opened": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "evidence_tier": response["evidence_tier"],
        "external_attestations_verified": False,
    }
    cell_seal = {
        **cell_body,
        "cell_response_seal_sha256": canonical_sha256(cell_body),
    }

    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError("--output-dir must be new; reuse is forbidden")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    response_path = output_dir / f"response-{response['response_receipt_sha256']}.json"
    cell_path = (
        output_dir / f"cell-response-{cell_seal['cell_response_seal_sha256']}.json"
    )
    _exclusive_write_json(response_path, response)
    _exclusive_write_json(cell_path, cell_seal)
    print(
        json.dumps(
            {
                "status": "DOJO_AI_RESPONSE_SEALED",
                "cell_id": assignment["cell_id"],
                "response_receipt_sha256": response["response_receipt_sha256"],
                "cell_response_seal_sha256": cell_seal["cell_response_seal_sha256"],
                "response_path": str(response_path),
                "cell_response_path": str(cell_path),
                "answer_key_opened": False,
                "live_permission": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_request_receipt(value: Mapping[str, Any]) -> None:
    claimed = value.get("request_receipt_sha256")
    body = {key: item for key, item in value.items() if key != "request_receipt_sha256"}
    if not isinstance(claimed, str) or claimed != canonical_sha256(body):
        raise ValueError("request receipt seal is invalid")
    if (
        value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
    ):
        raise ValueError("request receipt violates offline-only boundary")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_or_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())

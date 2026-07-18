#!/usr/bin/env python3
"""Build one offline, calendar-date-removed DOJO AI-discretion trial bundle."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.dojo_ai_discretion import (
    DIAGNOSTIC_TIER,
    FIXED_RESPONSE_SCHEMA,
    RESPONSE_CONTRACT,
    build_capability_manifest,
    build_day_packet,
    canonical_sha256,
    prelock_prompt,
    prelock_scorer,
    seal_model_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-lineage", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--locked-at-utc")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    locked_at = _parse_or_now(args.locked_at_utc)
    source = _read_object(args.source)
    prompt_text = args.prompt_file.read_text(encoding="utf-8")

    capability = build_capability_manifest(
        context_id=args.context_id,
        generated_at_utc=locked_at,
    )
    prompt = prelock_prompt(
        prompt_text,
        variant_id=args.variant_id,
        locked_at_utc=locked_at,
    )
    scorer = prelock_scorer(locked_at_utc=locked_at)
    model = seal_model_manifest(
        model_name=args.model_name,
        model_version=args.model_version,
        model_lineage=args.model_lineage,
        reasoning_effort=args.reasoning_effort,
        context_id=args.context_id,
        capability_manifest=capability,
        locked_at_utc=locked_at,
    )
    packet = build_day_packet(
        source,
        prompt_lock=prompt,
        capability_manifest=capability,
        scorer_lock=scorer,
    )

    request_body = {
        "contract": "QR_DOJO_AI_DISCRETION_REQUEST_RECEIPT_V1",
        "schema_version": 1,
        "validity_status": packet["validity_status"],
        "trial_id": packet["trial_id"],
        "packet_sha256": packet["packet_sha256"],
        "prompt_lock_sha256": prompt["prompt_lock_sha256"],
        "prompt_sha256": prompt["prompt_sha256"],
        "model_sha256": model["model_sha256"],
        "declared_model_lineage": model["declared_model_lineage"],
        "model_lineage_attestation_status": model["model_lineage_attestation_status"],
        "capability_manifest_sha256": capability["capability_manifest_sha256"],
        "scorer_lock_sha256": scorer["scorer_lock_sha256"],
        "scorer_sha256": scorer["scorer_sha256"],
        "response_contract": RESPONSE_CONTRACT,
        "fixed_output_schema": FIXED_RESPONSE_SCHEMA,
        "declared_answer_key_must_be_physically_absent": True,
        "declared_fresh_context_required": True,
        "claims_are_self_attested": True,
        "model_api_invoked": False,
        "external_send_allowed": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "external_attestations_verified": False,
        "attestation_gap_codes": packet["attestation_gap_codes"],
    }
    request_receipt = {
        **request_body,
        "request_receipt_sha256": canonical_sha256(request_body),
    }

    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(
            "--output-dir must be new; reused directories are forbidden"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    outputs = {
        f"capability-manifest-{capability['capability_manifest_sha256']}.json": capability,
        f"prompt-lock-{prompt['prompt_lock_sha256']}.json": prompt,
        f"scorer-lock-{scorer['scorer_lock_sha256']}.json": scorer,
        f"model-manifest-{model['model_sha256']}.json": model,
        f"packet-{packet['packet_sha256']}.json": packet,
        f"request-receipt-{request_receipt['request_receipt_sha256']}.json": request_receipt,
    }
    for name, value in outputs.items():
        _exclusive_write_json(output_dir / name, value)

    print(
        json.dumps(
            {
                "status": "DOJO_AI_PACKET_LOCKED",
                "trial_id": packet["trial_id"],
                "packet_sha256": packet["packet_sha256"],
                "request_receipt_sha256": request_receipt["request_receipt_sha256"],
                "output_dir": str(output_dir),
                "artifacts": {
                    name.split("-", 1)[0]: str(output_dir / name) for name in outputs
                },
                "evidence_tier": DIAGNOSTIC_TIER,
                "external_attestations_verified": False,
                "model_api_invoked": False,
                "live_permission": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _parse_or_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("--locked-at-utc must be an aware timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("--locked-at-utc must be an aware timestamp")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())

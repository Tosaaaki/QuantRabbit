#!/usr/bin/env python3
"""Score sealed DOJO responses offline; never invoke a model or live gateway."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.dojo_ai_discretion import (
    DIAGNOSTIC_TIER,
    assert_no_stale_positive_artifacts,
    canonical_sha256,
    score_pilot,
    score_sealed_response,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, action="append", required=True)
    parser.add_argument("--answer-key", type=Path, action="append", required=True)
    parser.add_argument("--packet", type=Path, action="append", required=True)
    parser.add_argument("--prompt-lock", type=Path, action="append", required=True)
    parser.add_argument("--model-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--capability-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--scorer-lock", type=Path, required=True)
    parser.add_argument(
        "--active-artifact",
        type=Path,
        action="append",
        default=[],
        help="Existing positive artifacts that must have a current valid seal.",
    )
    parser.add_argument(
        "--validity-registry",
        type=Path,
        help="Current sealed validity heads required for every active positive artifact.",
    )
    parser.add_argument("--opened-at-utc")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trial_counts = {
        len(args.response),
        len(args.answer_key),
        len(args.packet),
        len(args.prompt_lock),
        len(args.model_manifest),
        len(args.capability_manifest),
    }
    if len(trial_counts) != 1:
        raise ValueError("all per-trial artifact counts must match")
    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(
            "--output-dir must be new; reused directories are forbidden"
        )
    opened_at = _parse_or_now(args.opened_at_utc)
    scorer_lock = _read_object(args.scorer_lock)

    # Stale positives are rejected before any answer key can be opened.
    active_artifacts = [_read_object(path) for path in args.active_artifact]
    validity_registry = (
        _read_object(args.validity_registry) if args.validity_registry else None
    )
    assert_no_stale_positive_artifacts(
        active_artifacts,
        validity_registry=validity_registry,
    )

    scores: list[dict[str, Any]] = []
    trial_paths = zip(
        args.response,
        args.answer_key,
        args.packet,
        args.prompt_lock,
        args.model_manifest,
        args.capability_manifest,
    )
    for (
        response_path,
        answer_key_path,
        packet_path,
        prompt_path,
        model_path,
        capability_path,
    ) in trial_paths:
        response = _read_object(response_path)
        packet = _read_object(packet_path)
        prompt = _read_object(prompt_path)
        model = _read_object(model_path)
        capability = _read_object(capability_path)

        # This closure is intentionally lazy. score_sealed_response verifies the
        # response and scorer seals before it calls the loader.
        def load_answer_key(path: Path = answer_key_path) -> dict[str, Any]:
            return _read_object(path)

        scores.append(
            score_sealed_response(
                response,
                packet=packet,
                prompt_lock=prompt,
                model_manifest=model,
                capability_manifest=capability,
                scorer_lock=scorer_lock,
                answer_key_loader=load_answer_key,
                opened_at_utc=opened_at,
            )
        )

    pilot = score_pilot(scores)
    bundle_body = {
        "contract": "QR_DOJO_AI_DISCRETION_SCORE_BUNDLE_V1",
        "schema_version": 1,
        "validity_status": pilot["validity_status"],
        "generated_at_utc": opened_at.isoformat(),
        "scores": scores,
        "pilot": pilot,
        "active_artifact_count": len(active_artifacts),
        "answer_keys_opened_only_after_response_seals": True,
        "answer_key_open_order_attestation_status": "SELF_ATTESTED_UNVERIFIED",
        "model_lineage_handling": "DECLARED_CLUSTER_ONLY_NO_INDEPENDENCE_CLAIM",
        "model_api_invoked": False,
        "external_send_allowed": False,
        "read_only": True,
        "ai_order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "external_attestations_verified": False,
        "attestation_gap_codes": pilot["attestation_gap_codes"],
    }
    bundle = {**bundle_body, "bundle_sha256": canonical_sha256(bundle_body)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    output_paths: dict[str, str] = {}
    for score in scores:
        path = output_dir / f"score-{score['score_receipt_sha256']}.json"
        _exclusive_write_json(path, score)
        output_paths[f"score:{score['trial_id']}"] = str(path)
    pilot_path = output_dir / f"pilot-{pilot['pilot_score_sha256']}.json"
    _exclusive_write_json(pilot_path, pilot)
    output_paths["pilot"] = str(pilot_path)
    bundle_path = output_dir / f"score-bundle-{bundle['bundle_sha256']}.json"
    _exclusive_write_json(bundle_path, bundle)
    output_paths["bundle"] = str(bundle_path)
    print(
        json.dumps(
            {
                "status": pilot["validity_status"],
                "trial_count": pilot["trial_count"],
                "declared_lineage_cluster_count_diagnostic": pilot[
                    "declared_lineage_cluster_count_diagnostic"
                ],
                "pilot_score_sha256": pilot["pilot_score_sha256"],
                "bundle_sha256": bundle["bundle_sha256"],
                "artifacts": output_paths,
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
        raise ValueError("--opened-at-utc must be an aware timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("--opened-at-utc must be an aware timestamp")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())

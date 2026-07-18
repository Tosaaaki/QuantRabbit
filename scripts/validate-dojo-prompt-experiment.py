#!/usr/bin/env python3
"""Fail closed when the DOJO prompt preregistration drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REGISTRY_RELATIVE_PATH = Path("research/registries/dojo_prompt_experiment_v1.json")
EXPECTED_VARIANTS = {
    "A_FABLE_MINIMAL": "research/prompts/dojo_fable_minimal_v1.txt",
    "B_CALIBRATED_ABSTENTION": "research/prompts/dojo_calibrated_abstention_v1.txt",
    "C_STRUCTURAL_REGIME": "research/prompts/dojo_structural_regime_v1.txt",
}
EXPECTED_PROMPT_HASHES = {
    "A_FABLE_MINIMAL": "b20c37aff8be4282110a05457c5434e6dc690fd70aa533e06c69e2c50a30cccf",
    "B_CALIBRATED_ABSTENTION": "97b413e5d5ad978a7825d08e0cb390e8d23bbb72d88f9023a6e67abfc219c2c2",
    "C_STRUCTURAL_REGIME": "02a204ade67002fae04a556b5a7f466a07cef28001620d1b7e6024c07bbb4fd5",
}
EXPECTED_SCORER_POLICY_SHA256 = (
    "76eff0705971fdd149bdf4364b240e9c60d61ac8fb61542e80345472008b8496"
)
EXPECTED_RESPONSE_KEYS = [
    "trial_id",
    "action",
    "pair",
    "size",
    "confidence",
    "evidence_refs",
    "target_pips",
    "invalidation_pips",
    "strongest_counterargument",
    "abstain_reason",
]
EXPECTED_ACTIONS = ["FLAT", "LONG", "SHORT"]
EXPECTED_SIZES = ["NONE", "HALF", "FULL"]

DATE_PATTERNS = {
    "ISO date": re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
    "numeric date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})\b"),
    "named month": re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\b",
        re.IGNORECASE,
    ),
    "weekday": re.compile(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
}
RESULT_CUE_PATTERNS = {
    "answer key": re.compile(r"\banswer[\s_-]*key\b", re.IGNORECASE),
    "result cue": re.compile(r"\boutcomes?\b", re.IGNORECASE),
    "realized value": re.compile(r"\breali[sz]ed\b", re.IGNORECASE),
    "future cue": re.compile(r"\bfuture\b", re.IGNORECASE),
    "forward cue": re.compile(r"\bforward\b", re.IGNORECASE),
    "post-cutoff cue": re.compile(r"\bpost[\s_-]*cutoff\b", re.IGNORECASE),
    "exit price": re.compile(r"\bexit[\s_-]*price\b", re.IGNORECASE),
    "ground truth": re.compile(r"\bground[\s_-]*truth\b", re.IGNORECASE),
    "target label": re.compile(r"\btarget[\s_-]*label\b", re.IGNORECASE),
    "winning cue": re.compile(r"\b(?:winner|winning|win[\s_-]*rate)\b", re.IGNORECASE),
    "profit cue": re.compile(r"\bprofit(?:able|ability|s)?\b", re.IGNORECASE),
    "PnL cue": re.compile(r"\b(?:pnl|p\s*&\s*l|p\s*/\s*l)\b", re.IGNORECASE),
    "subsequent-price cue": re.compile(r"\bsubsequent[\s_-]*price\b", re.IGNORECASE),
}


class ValidationError(ValueError):
    """The preregistration is invalid or has drifted."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_registry(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load registry {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValidationError("registry root must be a JSON object")
    return loaded


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValidationError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _prompt_path(repo_root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValidationError(f"prompt path must be repository-relative: {relative}")
    resolved = (repo_root / relative_path).resolve()
    prompt_root = (repo_root / "research/prompts").resolve()
    try:
        resolved.relative_to(prompt_root)
    except ValueError as exc:
        raise ValidationError(f"prompt escapes research/prompts: {relative}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise ValidationError(f"prompt is missing or is not a regular file: {relative}")
    return resolved


def _schema_example(prompt: str, variant_id: str) -> dict[str, Any]:
    candidates = [
        line.strip() for line in prompt.splitlines() if line.strip().startswith("{")
    ]
    if len(candidates) != 1:
        raise ValidationError(
            f"{variant_id} must contain exactly one JSON schema example"
        )
    try:
        example = json.loads(
            candidates[0], object_pairs_hook=_object_without_duplicates
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"{variant_id} schema example is invalid JSON: {exc}"
        ) from exc
    if not isinstance(example, dict):
        raise ValidationError(f"{variant_id} schema example must be an object")
    return example


def _validate_prompt(prompt: str, variant_id: str) -> None:
    if not prompt.strip():
        raise ValidationError(f"{variant_id} prompt is empty")
    if len(prompt.encode("utf-8")) > 2_500:
        raise ValidationError(f"{variant_id} prompt is no longer concise")
    for label, pattern in DATE_PATTERNS.items():
        if pattern.search(prompt):
            raise ValidationError(f"{variant_id} contains date leakage ({label})")
    for label, pattern in RESULT_CUE_PATTERNS.items():
        if pattern.search(prompt):
            raise ValidationError(
                f"{variant_id} contains a forbidden result cue ({label})"
            )

    example = _schema_example(prompt, variant_id)
    _expect(list(example), EXPECTED_RESPONSE_KEYS, f"{variant_id} response keys")
    _expect(example["action"], "FLAT|LONG|SHORT", f"{variant_id} action enum")
    _expect(example["size"], "NONE|HALF|FULL", f"{variant_id} size enum")
    if "exactly one JSON object and no prose" not in prompt:
        raise ValidationError(
            f"{variant_id} does not enforce the fixed JSON-only response"
        )

    if variant_id == "A_FABLE_MINIMAL" and "Fable" not in prompt:
        raise ValidationError("A_FABLE_MINIMAL is no longer a Fable baseline")
    if variant_id == "B_CALIBRATED_ABSTENTION":
        for phrase in (
            "symmetric decision loss",
            "FLAT as an active choice",
            "strongest",
        ):
            if phrase not in prompt:
                raise ValidationError(
                    f"B_CALIBRATED_ABSTENTION lost required phrase: {phrase}"
                )
    if variant_id == "C_STRUCTURAL_REGIME":
        for phrase in ("trend, range, transition, or dislocation", "supply and demand"):
            if phrase not in prompt:
                raise ValidationError(
                    f"C_STRUCTURAL_REGIME lost required phrase: {phrase}"
                )


def _validate_variants(registry: dict[str, Any], repo_root: Path) -> None:
    variants = registry.get("variants")
    if not isinstance(variants, list):
        raise ValidationError("variants must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for raw_variant in variants:
        if not isinstance(raw_variant, dict):
            raise ValidationError("every variant must be an object")
        _expect(
            set(raw_variant),
            {"variant_id", "role", "prompt_path", "prompt_sha256"},
            "variant fields",
        )
        variant_id = raw_variant.get("variant_id")
        if not isinstance(variant_id, str) or variant_id in by_id:
            raise ValidationError(f"invalid or duplicate variant id: {variant_id!r}")
        by_id[variant_id] = raw_variant
    _expect(set(by_id), set(EXPECTED_VARIANTS), "registered variant ids")

    for variant_id, expected_relative in EXPECTED_VARIANTS.items():
        variant = by_id[variant_id]
        _expect(variant["prompt_path"], expected_relative, f"{variant_id} prompt path")
        path = _prompt_path(repo_root, expected_relative)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        _expect(variant["prompt_sha256"], digest, f"{variant_id} prompt SHA-256")
        try:
            prompt = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"{variant_id} prompt is not UTF-8") from exc
        _validate_prompt(prompt, variant_id)
        _expect(
            variant["prompt_sha256"],
            EXPECTED_PROMPT_HASHES[variant_id],
            f"{variant_id} locked prompt SHA-256",
        )


def _validate_protocol(registry: dict[str, Any]) -> None:
    _expect(
        registry.get("contract"), "DOJO_PROMPT_EXPERIMENT_PREREGISTRATION", "contract"
    )
    _expect(registry.get("schema_version"), 1, "schema version")
    _expect(registry.get("experiment_id"), "dojo-prompt-experiment-v1", "experiment id")
    _expect(registry.get("lock_status"), "SELF_ATTESTED_PRELOCK", "lock status")
    _expect(
        registry.get("evidence_tier"),
        "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC",
        "evidence tier",
    )
    locked_text = str(registry.get("locked_at_utc"))
    if locked_text.endswith("Z"):
        locked_text = locked_text[:-1] + "+00:00"
    try:
        locked = datetime.fromisoformat(locked_text)
    except ValueError as exc:
        raise ValidationError("locked_at_utc must be an ISO timestamp") from exc
    if locked.utcoffset() is None or locked.utcoffset().total_seconds() != 0:
        raise ValidationError("locked_at_utc must be UTC")

    response_schema = registry.get("response_schema", {})
    _expect(
        response_schema.get("exact_keys_in_order"),
        EXPECTED_RESPONSE_KEYS,
        "response keys",
    )
    _expect(response_schema.get("action_enum"), EXPECTED_ACTIONS, "action enum")
    _expect(response_schema.get("size_enum"), EXPECTED_SIZES, "size enum")
    _expect(
        response_schema.get("additional_properties"), False, "additional properties"
    )

    allocation = registry.get("allocation", {})
    phase_1 = allocation.get("phase_1_diagnostic", {})
    confirmation = allocation.get("future_confirmation", {})
    _expect(phase_1.get("blind_day_ranks_in_eligible_order"), "1-30", "phase-1 ranks")
    _expect(phase_1.get("unique_blind_days"), 30, "phase-1 day count")
    _expect(phase_1.get("cells_per_variant"), 30, "phase-1 cells per variant")
    _expect(phase_1.get("total_allocated_cells"), 90, "phase-1 cell count")
    _expect(
        phase_1.get("use"),
        "DIAGNOSTIC_ONLY_NO_SELECTION_NO_PROMPT_OR_SCORER_EDIT",
        "phase-1 use",
    )
    _expect(
        confirmation.get("blind_day_ranks_in_eligible_order"),
        "31-60",
        "confirmation ranks",
    )
    _expect(confirmation.get("unique_blind_days"), 30, "confirmation day count")
    _expect(confirmation.get("cells_per_variant"), 30, "confirmation cells per variant")
    _expect(confirmation.get("total_allocated_cells"), 90, "confirmation cell count")
    _expect(
        confirmation.get("use"),
        "UNTOUCHED_CONFIRMATION_ALL_THREE_LOCKED_VARIANTS",
        "confirmation use",
    )
    _expect(
        allocation.get("per_day_variant_allocation"),
        {variant_id: 1 for variant_id in EXPECTED_VARIANTS},
        "per-day variant allocation",
    )
    _expect(
        allocation.get("fresh_context_policy"),
        "ONE_NEW_STATELESS_MODEL_CONTEXT_PER_BLIND_DAY_PER_VARIANT",
        "fresh-context policy",
    )
    _expect(allocation.get("context_reuse_allowed"), False, "context reuse")
    _expect(
        allocation.get("cross_variant_message_history_allowed"),
        False,
        "cross-variant history",
    )
    _expect(allocation.get("total_unique_blind_days"), 60, "total day count")
    _expect(allocation.get("total_allocated_cells"), 180, "total cell count")

    lock_rules = registry.get("lock_rules", {})
    for key in (
        "prompt_edits_after_lock_allowed",
        "scorer_edits_after_lock_allowed",
        "allocation_edits_after_lock_allowed",
        "phase_1_can_select_or_drop_variants",
        "response_visibility_before_all_phase_cells_sealed",
        "answer_key_visibility_before_response_seal",
    ):
        _expect(lock_rules.get(key), False, f"lock rule {key}")

    score_policy = registry.get("score_policy")
    if not isinstance(score_policy, dict):
        raise ValidationError("score_policy must be an object")
    _expect(
        registry.get("scorer_policy_sha256"),
        canonical_sha256(score_policy),
        "scorer policy SHA-256",
    )
    _expect(
        registry.get("scorer_policy_sha256"),
        EXPECTED_SCORER_POLICY_SHA256,
        "locked scorer policy SHA-256",
    )
    _expect(score_policy.get("policy_id"), "DOJO_PROMPT_FIXED_SCORER_V1", "scorer id")
    _expect(
        score_policy.get("primary_metric"),
        "NET_LOG_GROWTH_ALL_ALLOCATED_CELLS",
        "primary metric",
    )
    _expect(
        score_policy.get("missing_or_schema_invalid_response"),
        "SYNTHETIC_FLAT_AND_RESPONSE_FAILURE_REMAINS_IN_DENOMINATOR",
        "missing-response policy",
    )
    _expect(
        score_policy.get("missing_response_action"), "FLAT", "missing-response action"
    )
    _expect(
        score_policy.get("missing_response_is_failure"),
        True,
        "missing-response failure flag",
    )
    _expect(
        score_policy.get("same_model_lineage_handling"),
        "CLUSTER_TOGETHER_NO_IID_AGENT_CLAIM",
        "lineage handling",
    )
    _expect(
        score_policy.get("independent_model_n"),
        "COUNT_EXTERNALLY_ATTESTED_DISTINCT_LINEAGES_OTHERWISE_ONE",
        "independent model count",
    )
    _expect(
        score_policy.get("phase_1_inference"),
        "DESCRIPTIVE_DIAGNOSTIC_ONLY",
        "phase-1 inference",
    )
    _expect(
        score_policy.get("future_confirmation_inference"),
        "PREDECLARED_PAIRED_CONTRASTS_ONLY",
        "confirmation inference",
    )
    _expect(
        score_policy.get("positive_result_grants_live_permission"),
        False,
        "scorer live grant",
    )

    guards = registry.get("execution_guards", {})
    for key in (
        "model_calls_performed_by_registration",
        "guarantee_monthly_3x",
        "guarantee_any_return",
        "live_trading_authorized",
        "broker_mutation_allowed",
    ):
        _expect(guards.get(key), False, f"execution guard {key}")
    _expect(guards.get("ai_order_authority"), "NONE", "AI order authority")
    _expect(guards.get("read_only_diagnostic"), True, "read-only diagnostic guard")

    attestations = registry.get("attestations", {})
    for key in (
        "external_capability_attestation_present",
        "provider_model_identity_and_lineage_attestation_present",
        "external_answer_key_absence_and_open_chronology_attestation_present",
        "external_registry_monotonicity_attestation_present",
        "tier_upgrade_allowed_now",
    ):
        _expect(attestations.get(key), False, f"attestation gate {key}")


def validate_registry(
    registry_path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    repo_root = repo_root.resolve()
    registry = _load_registry(registry_path)
    _validate_protocol(registry)
    _validate_variants(registry, repo_root)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / REGISTRY_RELATIVE_PATH,
    )
    args = parser.parse_args(argv)
    try:
        registry = validate_registry(args.registry)
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"VALID: {registry['experiment_id']} "
        f"({len(registry['variants'])} variants, {registry['evidence_tier']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

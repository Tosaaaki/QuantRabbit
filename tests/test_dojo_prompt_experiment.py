from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/validate-dojo-prompt-experiment.py"
REGISTRY_PATH = REPO_ROOT / "research/registries/dojo_prompt_experiment_v1.json"

SPEC = importlib.util.spec_from_file_location(
    "dojo_prompt_experiment_validator", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _copy_experiment(tmp_path: Path) -> tuple[Path, dict]:
    shutil.copytree(REPO_ROOT / "research/prompts", tmp_path / "research/prompts")
    target_registry = tmp_path / "research/registries/dojo_prompt_experiment_v1.json"
    target_registry.parent.mkdir(parents=True)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    target_registry.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return target_registry, registry


def _write_registry(path: Path, registry: dict) -> None:
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def _refresh_prompt_hash(tmp_path: Path, registry: dict, variant_id: str) -> None:
    variant = next(
        item for item in registry["variants"] if item["variant_id"] == variant_id
    )
    raw = (tmp_path / variant["prompt_path"]).read_bytes()
    variant["prompt_sha256"] = hashlib.sha256(raw).hexdigest()


def test_locked_prompt_experiment_validates() -> None:
    registry = VALIDATOR.validate_registry(REGISTRY_PATH, repo_root=REPO_ROOT)

    assert registry["evidence_tier"] == "SELF_ATTESTED_UNVERIFIED_DIAGNOSTIC"
    assert registry["execution_guards"]["guarantee_monthly_3x"] is False
    assert registry["execution_guards"]["live_trading_authorized"] is False


def test_validator_rejects_prompt_hash_drift(tmp_path: Path) -> None:
    target_registry, registry = _copy_experiment(tmp_path)
    prompt_path = tmp_path / registry["variants"][0]["prompt_path"]
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + "\nDrift.\n", encoding="utf-8"
    )

    with pytest.raises(VALIDATOR.ValidationError, match="prompt SHA-256"):
        VALIDATOR.validate_registry(target_registry, repo_root=tmp_path)


def test_validator_rejects_coordinated_prompt_rehash(tmp_path: Path) -> None:
    target_registry, registry = _copy_experiment(tmp_path)
    prompt_path = tmp_path / registry["variants"][0]["prompt_path"]
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8")
        + "\nUse packet evidence conservatively.\n",
        encoding="utf-8",
    )
    _refresh_prompt_hash(tmp_path, registry, "A_FABLE_MINIMAL")
    _write_registry(target_registry, registry)

    with pytest.raises(VALIDATOR.ValidationError, match="locked prompt SHA-256"):
        VALIDATOR.validate_registry(target_registry, repo_root=tmp_path)


def test_validator_rejects_unknown_variant(tmp_path: Path) -> None:
    target_registry, registry = _copy_experiment(tmp_path)
    registry["variants"][0]["variant_id"] = "A_UNKNOWN"
    _write_registry(target_registry, registry)

    with pytest.raises(VALIDATOR.ValidationError, match="registered variant ids"):
        VALIDATOR.validate_registry(target_registry, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("leaked_text", "error"),
    [
        ("Calendar marker: 2025-03-14.", "date leakage"),
        ("Use the future outcome as a hint.", "forbidden result cue"),
    ],
)
def test_validator_rejects_prompt_leakage(
    tmp_path: Path,
    leaked_text: str,
    error: str,
) -> None:
    target_registry, registry = _copy_experiment(tmp_path)
    prompt_path = tmp_path / registry["variants"][0]["prompt_path"]
    prompt_path.write_text(
        prompt_path.read_text(encoding="utf-8") + f"\n{leaked_text}\n",
        encoding="utf-8",
    )
    _refresh_prompt_hash(tmp_path, registry, "A_FABLE_MINIMAL")
    _write_registry(target_registry, registry)

    with pytest.raises(VALIDATOR.ValidationError, match=error):
        VALIDATOR.validate_registry(target_registry, repo_root=tmp_path)


def test_validator_rejects_scorer_drift(tmp_path: Path) -> None:
    target_registry, registry = _copy_experiment(tmp_path)
    registry["score_policy"]["missing_response_is_failure"] = False
    _write_registry(target_registry, registry)

    with pytest.raises(VALIDATOR.ValidationError, match="scorer policy SHA-256"):
        VALIDATOR.validate_registry(target_registry, repo_root=tmp_path)


def test_validator_rejects_coordinated_scorer_rehash(tmp_path: Path) -> None:
    target_registry, registry = _copy_experiment(tmp_path)
    registry["score_policy"]["missing_response_is_failure"] = False
    registry["scorer_policy_sha256"] = VALIDATOR.canonical_sha256(
        registry["score_policy"]
    )
    _write_registry(target_registry, registry)

    with pytest.raises(VALIDATOR.ValidationError, match="locked scorer policy SHA-256"):
        VALIDATOR.validate_registry(target_registry, repo_root=tmp_path)

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/run-paper-champion-challenger.py").resolve()
SPEC = importlib.util.spec_from_file_location("paper_cc_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

CONFIG_PATH = Path("config/paper_champion_challenger_pullback_20260728_v1.json")


def test_shared_paper_experiment_matrix_is_valid_and_paper_only() -> None:
    config = RUNNER.load_experiment(CONFIG_PATH)
    assert len(config["lanes"]) == 6
    assert {lane["role"] for lane in config["lanes"]} == {
        "CHAMPION",
        "AI_INVENTORY",
        "CHALLENGER",
    }
    assert config["authority"]["order_authority"] == "NONE"
    assert config["dojo_dependency"] == "NONE"


def test_shared_paper_experiment_fails_closed_on_authority_or_matrix(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    live = copy.deepcopy(config)
    live["authority"]["live_permission"] = True
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live))
    with pytest.raises(RUNNER.ExperimentConfigError, match="authority"):
        RUNNER.load_experiment(live_path)

    incomplete = copy.deepcopy(config)
    incomplete["lanes"].pop()
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps(incomplete))
    with pytest.raises(RUNNER.ExperimentConfigError, match="exactly"):
        RUNNER.load_experiment(incomplete_path)

    mismatched_candidate = copy.deepcopy(config)
    mismatched_candidate["candidate"]["shared_feed_contract_sha256"] = "0" * 64
    mismatched_candidate["candidate"]["candidate_hash"] = RUNNER.candidate_hash(
        mismatched_candidate["candidate"]
    )
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(mismatched_candidate))
    with pytest.raises(RUNNER.ExperimentConfigError, match="shared-feed"):
        RUNNER.load_experiment(mismatch_path)

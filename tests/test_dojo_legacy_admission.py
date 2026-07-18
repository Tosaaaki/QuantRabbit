from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_goal_board import canonical_sha256
from quant_rabbit.dojo_legacy_admission import (
    LegacyAdmissionError,
    admit_legacy_positive_sources,
    load_current_goal_board,
    regular_file_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _eligible_source(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "contract": "QR_TEST_PROSPECTIVE_PROOF_V1",
        "diagnostic_only": False,
        "historical_only": False,
        "forward_proof_eligible": True,
        "promotion_allowed": True,
        "live_permission": False,
        "order_authority": "NONE",
    }
    value.update(overrides)
    return value


def _sealed_board(
    registry: Path,
    source_rows: list[tuple[str, str]],
    *,
    promotion_possible: bool = True,
) -> tuple[Path, dict[str, Any]]:
    lane_evaluations = []
    for index, (source_name, source_sha) in enumerate(source_rows):
        lane_evaluations.append(
            {
                "lane_id": f"lane_{index}_{source_name}",
                "declared_status": "EDGE_PROVEN",
                "edge_status": "EDGE_PROVEN",
                "evidence_verification": {
                    "trusted": True,
                    "status": "VERIFIED",
                    "blocker": None,
                    "actual_sha256": source_sha,
                    "expected_sha256": source_sha,
                    "contract": "QR_TEST_PROSPECTIVE_PROOF_V1",
                },
            }
        )
    body = {
        "contract": "QR_DOJO_GOAL_BOARD_V2",
        "schema_version": 2,
        "proof_admission": {
            "promotion_possible": promotion_possible,
            "self_asserted_json_can_promote": False,
            "trusted_proof_contracts": ["QR_TEST_PROSPECTIVE_PROOF_V1"],
        },
        "lane_evaluations": lane_evaluations,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
        "edge_status": "EDGE_PROVEN" if promotion_possible else "HYPOTHESIS",
        "goal_status": "3X_NOT_REACHABLE",
    }
    board = {**body, "board_sha256": canonical_sha256(body)}
    path = registry / f"dojo_goal_board_20260719_{board['board_sha256']}.json"
    _write_json(path, board)
    return path, board


def test_admits_only_registered_eligible_positive_sources(tmp_path: Path) -> None:
    sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    source_rows = []
    for name in ("survivor_lock", "validation_replication"):
        path = tmp_path / "data" / f"{name}.json"
        value = _eligible_source(artifact_id=name)
        _write_json(path, value)
        sources[name] = (path, value)
        source_rows.append((name, regular_file_sha256(path)))
    board_path, board = _sealed_board(tmp_path / "registry", source_rows)

    receipt = admit_legacy_positive_sources(
        board_path=board_path,
        board=board,
        positive_sources=sources,
    )

    assert receipt["contract"] == "QR_DOJO_LEGACY_POSITIVE_ADMISSION_V1"
    assert set(receipt["registrations"]) == set(sources)
    assert receipt["live_permission"] is False
    assert receipt["order_authority"] == "NONE"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"diagnostic_only": True}, "diagnostic_only=false"),
        ({"forward_proof_eligible": False}, "forward_proof_eligible=true"),
        ({"promotion_allowed": False}, "promotion_allowed=true"),
        ({"historical_only": True}, "historical_only=false"),
    ],
)
def test_rejects_ineligible_positive_source(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    path = tmp_path / "source.json"
    value = _eligible_source(**override)
    _write_json(path, value)
    board_path, board = _sealed_board(
        tmp_path / "registry", [("source", regular_file_sha256(path))]
    )

    with pytest.raises(LegacyAdmissionError, match=message):
        admit_legacy_positive_sources(
            board_path=board_path,
            board=board,
            positive_sources={"source": (path, value)},
        )


def test_rejects_unregistered_positive_source(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    value = _eligible_source()
    _write_json(path, value)
    board_path, board = _sealed_board(tmp_path / "registry", [("different", "0" * 64)])

    with pytest.raises(LegacyAdmissionError, match="not uniquely registered"):
        admit_legacy_positive_sources(
            board_path=board_path,
            board=board,
            positive_sources={"source": (path, value)},
        )


def test_rejects_source_changed_after_parse(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    value = _eligible_source()
    _write_json(path, value)
    source_sha = regular_file_sha256(path)
    board_path, board = _sealed_board(tmp_path / "registry", [("source", source_sha)])

    with pytest.raises(LegacyAdmissionError, match="changed after"):
        admit_legacy_positive_sources(
            board_path=board_path,
            board=board,
            positive_sources={"source": (path, value)},
            loaded_source_sha256={"source": "0" * 64},
        )


def test_rejects_goal_board_changed_after_load(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    value = _eligible_source()
    _write_json(path, value)
    board_path, board = _sealed_board(
        tmp_path / "registry", [("source", regular_file_sha256(path))]
    )
    changed = dict(board)
    changed["goal_status"] = "GOAL_COMPATIBLE"
    changed_body = dict(changed)
    changed_body.pop("board_sha256")
    changed["board_sha256"] = canonical_sha256(changed_body)
    replacement = board_path.with_name(
        f"dojo_goal_board_20260719_{changed['board_sha256']}.json"
    )
    board_path.unlink()
    _write_json(replacement, changed)

    with pytest.raises(LegacyAdmissionError, match="changed after"):
        admit_legacy_positive_sources(
            board_path=board_path,
            board=board,
            positive_sources={"source": (path, value)},
        )


def test_one_dojo_lane_cannot_admit_two_positive_sources(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    value = _eligible_source()
    _write_json(first_path, value)
    _write_json(second_path, value)
    source_sha = regular_file_sha256(first_path)
    board_path, board = _sealed_board(
        tmp_path / "registry", [("shared", source_sha)]
    )

    with pytest.raises(LegacyAdmissionError, match="reuses a DOJO lane"):
        admit_legacy_positive_sources(
            board_path=board_path,
            board=board,
            positive_sources={
                "first": (first_path, value),
                "second": (second_path, value),
            },
        )


def test_rejects_board_without_promotion_admission(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    value = _eligible_source()
    _write_json(path, value)
    board_path, board = _sealed_board(
        tmp_path / "registry",
        [("source", regular_file_sha256(path))],
        promotion_possible=False,
    )

    with pytest.raises(LegacyAdmissionError, match="does not permit"):
        admit_legacy_positive_sources(
            board_path=board_path,
            board=board,
            positive_sources={"source": (path, value)},
        )


def test_current_board_loader_rejects_tamper_and_ambiguity(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    board_path, board = _sealed_board(registry, [])
    loaded_path, loaded = load_current_goal_board(registry)
    assert loaded_path == board_path
    assert loaded == board

    second = dict(board)
    second["goal_status"] = "GOAL_COMPATIBLE"
    body = dict(second)
    body.pop("board_sha256")
    second["board_sha256"] = canonical_sha256(body)
    second_path = registry / f"dojo_goal_board_20260720_{second['board_sha256']}.json"
    _write_json(second_path, second)
    with pytest.raises(LegacyAdmissionError, match="exactly one"):
        load_current_goal_board(registry)

    second_path.unlink()
    board["goal_status"] = "GOAL_COMPATIBLE"
    _write_json(board_path, board)
    with pytest.raises(LegacyAdmissionError, match="canonical SHA-256"):
        load_current_goal_board(registry)


def test_adopted_stack_cli_blocks_current_unadmitted_sources(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    digest_keys = {
        "survivor_lock": "lock_sha256",
        "validation_replication": "evaluation_sha256",
        "throttle_comparison": "comparison_sha256",
        "overlay_sweep": "sweep_sha256",
        "nanpin_margin_feasible": "rehearsal_sha256",
        "nanpin_true_concurrency": "rehearsal_sha256",
        "monthly_distribution": "distribution_sha256",
        "all_weather_attribution": "attribution_sha256",
        "cell_gating": "rehearsal_sha256",
        "lane_addition": "combination_sha256",
    }
    filenames = {
        "survivor_lock": "adaptive_exact_s5_train_lock_v1.json",
        "validation_replication": "adaptive_exact_s5_validation_replication_v1.json",
        "throttle_comparison": "throttle_mode_comparison_v1.json",
        "overlay_sweep": "overlay_sweep_lab_v1.json",
        "nanpin_margin_feasible": "nanpin_margin_feasible_v1.json",
        "nanpin_true_concurrency": "nanpin_true_concurrency_v1.json",
        "monthly_distribution": "monthly_multiple_distribution_v1.json",
        "all_weather_attribution": "all_weather_attribution_v1.json",
        "cell_gating": "regime_cell_gating_rehearsal_v1.json",
        "lane_addition": "lane_addition_combination_v1.json",
    }
    for name, filename in filenames.items():
        body = _eligible_source()
        body["spec"] = {"spec_id": "TEST"}
        body["evaluation_policy"] = {}
        body["distribution_rows"] = []
        digest_key = digest_keys[name]
        _write_json(data_dir / filename, {**body, digest_key: canonical_sha256(body)})
    registry = tmp_path / "registry"
    _sealed_board(registry, [], promotion_possible=False)
    output = tmp_path / "adopted.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build-adopted-stack-tuning-contract.py"),
            "--data-dir",
            str(data_dir),
            "--dojo-registry",
            str(registry),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "does not permit evidence promotion" in result.stderr
    assert not output.exists()


def test_system_map_uses_current_dojo_board_as_entry_point(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    output = tmp_path / "system-map.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build-system-map.py"),
            "--repo-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["contract"] == "QR_SYSTEM_MAP_V2"
    assert value["single_entry_point_for_codex"].startswith(
        "research/registries/dojo_goal_board_"
    )
    assert value["legacy_positive_artifacts"]["status"] == (
        "NOT_AN_ENTRY_POINT_REQUIRES_CURRENT_DOJO_ADMISSION"
    )
    assert value["current_dojo_validity"]["promotion_possible"] is False

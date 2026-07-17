from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_NAMESPACE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build-profit-path-feasibility-board.py"
    )
)
build_board = _NAMESPACE["build_board"]
daily_return_fraction = _NAMESPACE["daily_return_fraction"]
required_pips_per_day = _NAMESPACE["required_pips_per_day"]
_canonical_sha = _NAMESPACE["_canonical_sha"]


def _sealed(body: dict, digest_key: str) -> dict:
    return {**body, digest_key: _canonical_sha(body)}


def _artifacts() -> tuple[dict, dict, dict]:
    lock = _sealed(
        {"spec": {"spec_id": "XS-test"}, "train_metrics": {"net_pips": 1000.0}},
        "lock_sha256",
    )
    validation = _sealed(
        {
            "lock_sha256": lock["lock_sha256"],
            "independent_validation_claim_allowed": False,
            "metrics": {
                "active_days": 10,
                "net_pips": 970.0,
                "stressed_net_pips": 870.0,
            },
        },
        "evaluation_sha256",
    )
    prospective = _sealed(
        {"shadow_lock_sha256": lock["lock_sha256"]}, "prospective_lock_sha256"
    )
    return lock, validation, prospective


def test_sizing_math_matches_closed_form() -> None:
    assert daily_return_fraction(97.0, 25.0, concurrent_positions=12) == pytest.approx(
        97.0 * 0.0001 * 25.0 / 12.0
    )
    required = required_pips_per_day(leverage=25.0, days=30, concurrent_positions=12)
    daily = daily_return_fraction(required, 25.0, concurrent_positions=12)
    assert (1.0 + daily) ** 30 == pytest.approx(4.0)


def test_board_seals_gap_and_forbids_goal_claims() -> None:
    lock, validation, prospective = _artifacts()

    board = build_board(lock, validation, prospective)

    body = {key: value for key, value in board.items() if key != "board_sha256"}
    assert board["board_sha256"] == _canonical_sha(body)
    assert board["four_x_supported_by_proven_evidence"] is False
    assert board["monthly_4x_claim_allowed"] is False
    assert board["unproven_rows_not_citable_as_goal_evidence"] is True
    assert board["proven_lane_count"] == 0
    assert board["order_authority"] == "NONE"
    assert all(
        row["citable_for_goal_claims"] is False for row in board["scenarios"]
    )
    survivor = board["lanes"][0]
    assert survivor["pips_per_day"] == pytest.approx(97.0)
    expected_gap = required_pips_per_day(
        leverage=25.0, days=30, concurrent_positions=12
    ) - 97.0
    assert board["four_x_gap_pips_per_day_at_max_leverage"] == pytest.approx(
        expected_gap, abs=1e-6
    )


def test_board_refuses_tampered_or_unbound_artifacts() -> None:
    lock, validation, prospective = _artifacts()

    tampered = dict(validation)
    tampered["metrics"] = {**validation["metrics"], "net_pips": 99_999.0}
    with pytest.raises(ValueError, match="digest"):
        build_board(lock, tampered, prospective)

    foreign_lock = _sealed(
        {"spec": {"spec_id": "XS-other"}, "train_metrics": {}}, "lock_sha256"
    )
    with pytest.raises(ValueError, match="not bound"):
        build_board(foreign_lock, validation, prospective)

    claiming = dict(validation)
    claiming["independent_validation_claim_allowed"] = True
    claiming = _sealed(
        {k: v for k, v in claiming.items() if k != "evaluation_sha256"},
        "evaluation_sha256",
    )
    with pytest.raises(ValueError, match="independence"):
        build_board(lock, claiming, prospective)

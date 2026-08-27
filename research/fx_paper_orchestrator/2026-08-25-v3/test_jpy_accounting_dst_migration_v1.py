from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import build_jpy_accounting_dst_migration_v1 as migration


ROOT = Path(__file__).resolve().parent


def load_jsonl(relative: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
        if line
    ]


@pytest.fixture(scope="module")
def payload() -> dict:
    return migration.validate(ROOT)


def test_migration_evidence_is_reproducible_non_strategy_and_paper_only(payload: dict) -> None:
    assert payload["classification"] == (
        "NON_STRATEGY_RUNTIME_ACCOUNTING_AND_CHRONOLOGY_MIGRATION"
    )
    assert payload["authority"] == migration.AUTHORITY
    assert payload["holdout_state"] == "UNOPENED"
    assert payload["external_orders"] == 0
    assert payload["official_strategy_run_performed"] is False
    assert payload["historical_seal_rewritten"] is False
    assert payload["diagnostic_reused_as_official_seal"] is False
    assert payload["strategy_adoption_authorized"] is False
    assert payload["profit_gate_pass_inferred"] is False
    assert payload["audit_sha256"] == migration.embedded_hash(payload, "audit_sha256")


def test_independent_reference_and_formal_diagnostic_hashes_are_fixed(payload: dict) -> None:
    fixtures = load_jsonl(payload["reference_fixture_path"])
    diagnostics = load_jsonl(payload["diagnostic_rows_path"])
    assert payload["reference_fixture_count"] == 4
    assert payload["reference_fixture_parity_passed"] is True
    assert all(row["parity"] for row in fixtures)
    assert all(row["account_currency_midpoint_conversion_used"] is False for row in fixtures)
    assert len(diagnostics) == (35 + 77 + 78) * 3
    assert payload["diagnostic_rows"] == len(diagnostics)
    assert migration.sha256_file(ROOT / payload["reference_fixture_path"]) == payload[
        "reference_fixture_file_sha256"
    ]
    assert migration.sha256_file(ROOT / payload["diagnostic_rows_path"]) == payload[
        "diagnostic_rows_file_sha256"
    ]


def test_formal_sign_aware_bbo_diagnostic_does_not_reveal_hidden_two_x(payload: dict) -> None:
    expected = {
        "V38": {
            "RAW_SIGNAL": 1.0080641889154207,
            "EXECUTABLE_BASE": 1.0053391996349363,
            "ADVERSE_STRESS": 1.0033267961123333,
        },
        "V40": {
            "RAW_SIGNAL": 1.0060651236461997,
            "EXECUTABLE_BASE": 1.0034844103115372,
            "ADVERSE_STRESS": 1.0014629916875744,
        },
        "V41": {
            "RAW_SIGNAL": 0.9970513875633013,
            "EXECUTABLE_BASE": 0.9944700913233082,
            "ADVERSE_STRESS": 0.9924441896177426,
        },
    }
    for cycle, arms in expected.items():
        summary = payload["sealed_plan_accounting_diagnostic"][cycle]
        assert summary["same_explicit_units_all_arms"] is True
        assert summary["terminal_open_inventory"] == 0
        assert summary["terminal_inventory_mtm_jpy"] == 0.0
        assert summary["hidden_2x_revealed"] is False
        for arm, expected_multiple in arms.items():
            observed = summary["arms"][arm]
            assert math.isclose(
                observed["formal_sign_aware_bbo_fixed_notional_multiple"],
                expected_multiple,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            assert observed["formal_below_2x"] is True
            assert observed["diagnostic_only"] is True
            assert observed["reusable_as_official_seal"] is False


def test_cost_arms_share_each_episode_units_and_leave_terminal_mtm_zero(payload: dict) -> None:
    rows = load_jsonl(payload["diagnostic_rows_path"])
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["episode_id"], []).append(row)
        assert row["sealed_signal_or_action_changed"] is False
        assert row["terminal_inventory_mtm_jpy"] == 0.0
    assert len(groups) == 35 + 77 + 78
    for group in groups.values():
        assert {row["scenario"] for row in group} == {
            "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS",
        }
        assert len({row["units"] for row in group}) == 1
        assert len({tuple(row["source_signal_ids"]) for row in group}) == 1


def test_dst_foundation_is_not_edge_and_existing_v42_work_order_is_unchanged(
    payload: dict,
) -> None:
    dst = payload["dst_chronology"]
    assert dst["classification"] == "COMMON_CHRONOLOGY_FOUNDATION_NOT_NEW_EDGE"
    assert dst["fixed_utc_hour_used_as_edge_definition"] is False
    assert dst["winter"]["utc_offset_seconds"] == 0
    assert dst["summer"]["utc_offset_seconds"] == 3600
    assert payload["dst_is_revenue_edge"] is False
    assert payload["v42_dst_only"] == "NO_GO"
    assert payload["current_official_v42_execution_authorized"] is False
    work_order = "evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json"
    assert payload["protected_historical_artifact_hashes"][work_order] == (
        "29d541646f57efffe543007d94ce0958a2fa3cc68180cf521101886a8b09b524"
    )

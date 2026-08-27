from __future__ import annotations

import json
import math
from pathlib import Path

import derived_pair_audit_runner_v1 as audit


ROOT = Path(__file__).resolve().parent


def load_metrics() -> list[dict]:
    return audit.load_jsonl(ROOT / audit.METRICS_PATH)


def test_readback_has_thirteen_valid_seals_six_raw_streams_and_excludes_failures() -> None:
    payload = audit.validate(ROOT)
    dedupe = payload["deduplication"]
    assert dedupe["valid_sealed_cycle_count"] == 13
    assert dedupe["unique_signal_id_set_count"] == 6
    assert dedupe["cycle_count_is_independent_trial_count"] is False
    assert dedupe["pair_signal_id_dedupe"] == audit.EXPECTED_DEDUPE
    failed = {item["cycle_id"]: item for item in payload["failed_and_invalid_cycles"]}
    assert set(failed) == {"V26", "V32", "V34", "V36"}
    assert all(item["metrics_admissible"] is False for item in failed.values())
    assert failed["V34"]["official_seal_exists"] is False


def test_every_pair_reconstruction_reconciles_to_sealed_portfolio() -> None:
    payload = audit.validate(ROOT)
    assert len(payload["portfolio_reconciliation"]) == 13 * 3
    for period in payload["portfolio_reconciliation"].values():
        assert set(period) == set(audit.ARMS)
        for arm in period.values():
            assert arm["passed"] is True
            assert arm["absolute_difference"] <= 1e-12
    assert payload["pair_metrics_rows"] == 13 * 3 * 7


def test_v25_pair_contributions_and_v27_selection_match_direct_readback() -> None:
    payload = audit.validate(ROOT)
    direct = payload["direct_pair_checks"]
    for pair, expected in audit.EXPECTED_V25_WALK_FORWARD.items():
        actual = direct["V25"][pair]
        assert actual["executed_episodes"] == expected["episodes"]
        assert math.isclose(actual["direction_accuracy"], expected["direction_accuracy"],
                            rel_tol=0.0, abs_tol=1e-15)
        for arm, expected_jpy in expected["jpy"].items():
            assert math.isclose(
                actual["jpy_contribution_legacy_sealed_convention"][arm],
                expected_jpy, rel_tol=0.0, abs_tol=0.0051,
            )
    assert direct["V27"]["AUD_USD"]["executed_episodes"] == 0
    assert direct["V27"]["EUR_USD"]["executed_episodes"] == 1
    assert direct["V27"]["USD_JPY"]["executed_episodes"] == 29
    assert direct["V27"]["EUR_USD"]["base_jpy"] > 0
    assert direct["V27"]["EUR_USD"]["adverse_jpy"] > 0


def test_pair_metrics_do_not_invent_adjusted_n_eff_or_new_accounting() -> None:
    payload = audit.validate(ROOT)
    for row in load_metrics():
        assert row["N_eff"]["autocorrelation_adjusted"] is None
        assert row["N_eff"]["common_currency_time_cluster_adjusted"] is None
        assert row["sealed_result_reclassified"] is False
        assert row["accounting_classification"].startswith("DERIVED_USING_LEGACY")
        for arm in audit.ARMS:
            assert row["arms"][arm]["terminal_open_inventory"] == 0
            assert row["arms"][arm]["terminal_inventory_mtm"] == 0.0
    accounting = payload["accounting_convention"]
    assert accounting["quote_to_jpy_conversion_present"] is False
    assert accounting["short_fixed_notional_linear_pnl_present"] is False
    assert accounting["migration_required_before_next_official_strategy_run"] is True


def test_completed_range_and_oracle_ceiling_exact_readback() -> None:
    payload = audit.validate(ROOT)
    evidence = payload["daily_range_and_oracle_feasibility"]
    for month, pairs in audit.EXPECTED_RANGE.items():
        actual = evidence["completed_daily_range"][month]
        for pair, (days, mean, median) in pairs.items():
            item = actual["pairs"][pair]
            assert item["eligible_utc_days"] == days
            assert math.isclose(item["mean_daily_range_pips"], mean,
                                rel_tol=0.0, abs_tol=1e-12)
            assert math.isclose(item["median_daily_range_pips"], median,
                                rel_tol=0.0, abs_tol=1e-12)
        for gross, expected in audit.EXPECTED_ORACLE_CAPTURE_PERCENT[month].items():
            solution = actual["oracle_capture_solutions"][gross]
            assert math.isclose(solution["required_daily_high_low_capture_percent"], expected,
                                rel_tol=0.0, abs_tol=1e-10)
            assert solution["strategy_evidence"] is False
    assert evidence["current_1x_required_capture_exceeds_full_daily_range"] is True
    assert evidence["gross_cap_change_authorized"] is False


def test_holdout_authority_and_protected_v42_work_order_remain_unchanged() -> None:
    payload = audit.validate(ROOT)
    assert payload["holdout_state"] == "UNOPENED"
    assert payload["official_strategy_run_performed"] is False
    assert payload["profit_gate_pass_inferred"] is False
    assert payload["strategy_adoption_authorized"] is False
    assert payload["authority"] == audit.AUTHORITY
    v42 = "evidence/orchestrator_state_v2/next_hypothesis_work_order_v42.json"
    assert payload["historical_artifact_hashes"]["after"][v42] == (
        "29d541646f57efffe543007d94ce0958a2fa3cc68180cf521101886a8b09b524"
    )


def test_future_geometric_gate_formula_is_fixed_but_thresholds_block_execution() -> None:
    policy = json.loads((ROOT / "PROFIT_GATE_CAUSE_FEASIBILITY_POLICY_V1.json").read_text())
    gate = policy["future_geometric_mean_profit_gate"]
    geometric = math.exp(statistics_mean := (
        math.log(gate["example"]["first_month_multiple"])
        + math.log(gate["example"]["required_second_month_multiple_for_two_month_G_2"])
    ) / 2.0)
    assert statistics_mean > 0
    assert math.isclose(geometric, 2.0, rel_tol=0.0, abs_tol=1e-15)
    for field in (
        "near_target_tolerance", "evaluation_month_count", "worst_month_floor",
        "maximum_drawdown_guard", "margin_guard", "ruin_guard",
    ):
        assert gate[field] is None
    assert gate["may_register_or_execute_future_profit_gate"] is False
    assert gate["legacy_month_by_month_gate_results_may_not_be_reclassified"] is True

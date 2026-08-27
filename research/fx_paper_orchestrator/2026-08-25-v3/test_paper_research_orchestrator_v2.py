from __future__ import annotations

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import paper_research_orchestrator_v2 as orchestrator


ROOT = Path(__file__).resolve().parent


def period_metrics(equity: float = 1.01, signals: int = 1) -> dict:
    return {
        "raw_diagnostics": {"signals": signals},
        **{
            arm: {"source_signals": signals, "terminal_open_inventory": 0, "equity_multiple": equity}
            for arm in orchestrator.ARMS
        },
    }


def synthetic_cycle() -> dict:
    return {
        "cycle_id": "V25",
        "source_contract": {"files": {}, "manifest_sha256": orchestrator.hashlib.sha256(
            orchestrator.canonical_bytes({})).hexdigest()},
        "inventory_contract": {
            "fixed_pair_sleeve": 1 / 7,
            "rule_max_gross_leverage": 1 / 7,
            "gross_leverage_cap": 1.0,
            "currency_abs_exposure_cap": 1.0,
            "finite_max_age_seconds": 21600,
        },
        "evaluation_contract": {
            "full_comparable_months": ["MONTH_2026_05", "MONTH_2026_06"],
            "holdout": {"state": "UNOPENED"},
        },
        "execution": {
            "argv": ["frozen.py", "--input-root", "/paper"],
            "pythonpath": [],
            "timeout_seconds": 10,
            "result": "evidence/run/result.json",
            "ledger": "evidence/run/ledger.jsonl",
        },
        "preregistration_sha256": "p" * 64,
        "script_sha256": "s" * 64,
        "test_sha256": "t" * 64,
    }


def write_synthetic_result(root: Path, cycle: dict, equity: float = 1.01) -> None:
    ledger = root / cycle["execution"]["ledger"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "signal_id": "S1",
        "pair": "EUR_USD",
        "utc_day": "2026-05-04",
        "decision_time": "2026-05-04T05:55:00.000000000Z",
        "fill_time": "2026-05-04T06:00:00.000000000Z",
        "exit_time": "2026-05-04T11:55:00.000000000Z",
        "direction": 1,
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "experiment": "SYNTHETIC",
        "evidence_class": "opened_development_not_future_holdout",
        "portfolio": {"gross_leverage_cap": 1.0, "weight_per_pair": 1 / 7},
        "raw_signals": 1,
        "effective_bet_days": 1,
        "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "proposal_ledger_sha256": orchestrator.sha256_file(ledger),
        "periods": {name: period_metrics(equity) for name in orchestrator.PERIODS},
        "source_audit": [],
        "development_admitted": False,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "live_authority": False,
        "external_orders": 0,
    }
    if cycle["cycle_id"] == "V27":
        result.update({
            "cycle_id": "V27",
            "experiment": "FX_CAUSAL_MIN_SPREAD_REPRESENTATIVE_V27",
            "runtime_compatibility_provenance": {
                "classification": "NON_STRATEGY_RUNTIME_COMPATIBILITY",
                "changed_strategy_variables": 0,
                "v26_rerun_permitted": False,
            },
        })
        for period in result["periods"].values():
            for metrics in (period[arm] for arm in orchestrator.ARMS):
                metrics["max_gross_exposure_nav"] = 1 / 7
                metrics["max_margin_requirement_jpy_at_1x"] = 200000 / 7
    result["result_sha256"] = orchestrator.embedded_hash(result, "result_sha256")
    result_path = root / cycle["execution"]["result"]
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


class RegistryAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = orchestrator.load_registry(ROOT, Path("PAPER_RESEARCH_CYCLE_REGISTRY_V2.json"))

    def test_real_v25_is_registered_and_frozen(self):
        cycle = self.registry["cycles"][0]
        self.assertEqual(cycle["cycle_id"], "V25")
        self.assertTrue(cycle["registered_before_official_execution"])
        self.assertEqual(cycle["hypothesis_contract"]["changed_variable_count"], 1)

    def test_v26_is_registered_once_from_sealed_v25_with_parent_raw_identity(self):
        self.assertEqual([cycle["cycle_id"] for cycle in self.registry["cycles"]], ["V25", "V26", "V27", "V28", "V29"])
        cycle = self.registry["cycles"][1]
        self.assertEqual(cycle["depends_on_cycle"], "V25")
        self.assertEqual(cycle["hypothesis_contract"]["changed_variable_count"], 1)
        self.assertEqual(cycle["signal_contract"]["raw_signal_source"], "SEALED_PARENT_V25_LEDGER")
        self.assertTrue(cycle["signal_contract"]["same_decision_timestamps"])
        self.assertTrue(cycle["signal_contract"]["same_execution_mask_all_arms"])

    def test_v27_is_new_cycle_with_same_unobserved_strategy_and_runtime_only_migration(self):
        cycle = self.registry["cycles"][2]
        prereg = json.loads((ROOT / cycle["preregistration"]).read_text())
        self.assertEqual(cycle["cycle_id"], "V27")
        self.assertEqual(cycle["depends_on_cycle"], "V25")
        self.assertEqual(cycle["hypothesis_contract"]["changed_variable_count"], 1)
        self.assertTrue(prereg["hypothesis_contract"]["same_unobserved_strategy_as_v26"])
        runtime = prereg["runtime_compatibility_provenance"]
        self.assertEqual(runtime["changed_strategy_variables"], 0)
        self.assertFalse(runtime["v26_rerun_permitted"])
        self.assertEqual(runtime["observed_corpus_nonzero_submicrosecond_count"], 0)

    def test_v28_is_one_training_only_causal_basket_hold_rule(self):
        cycle = self.registry["cycles"][3]
        prereg = json.loads((ROOT / cycle["preregistration"]).read_text())
        self.assertEqual(cycle["cycle_id"], "V28")
        self.assertEqual(cycle["depends_on_cycle"], "V27")
        self.assertEqual(cycle["hypothesis_contract"]["changed_variable_count"], 1)
        self.assertEqual(prereg["training_only_rule_selection"]["candidate_rules_compared"], 1)
        self.assertFalse(prereg["training_only_rule_selection"]["return_outcome_consulted"])
        self.assertFalse(prereg["training_only_rule_selection"]["cost_consulted"])
        self.assertEqual(cycle["inventory_contract"]["finite_max_age_seconds"], 345600)
        self.assertEqual(cycle["inventory_contract"]["rule_max_gross_leverage"], 1.0)

    def test_v29_is_one_training_only_cost_independent_consensus_release_rule(self):
        cycle = self.registry["cycles"][4]
        prereg = json.loads((ROOT / cycle["preregistration"]).read_text())
        self.assertEqual(cycle["cycle_id"], "V29")
        self.assertEqual(cycle["depends_on_cycle"], "V28")
        self.assertEqual(cycle["hypothesis_contract"]["changed_variable_count"], 1)
        selection = prereg["training_only_rule_selection"]
        self.assertEqual(selection["candidate_rules_compared"], 1)
        self.assertEqual(selection["selected_rule_structural_release_count"], 11)
        self.assertFalse(selection["return_outcome_consulted"])
        self.assertFalse(selection["cost_consulted"])
        rule = prereg["execution_rule"]
        self.assertEqual(rule["minimum_peer_signals"], 2)
        self.assertTrue(rule["unanimity_required"])
        self.assertEqual(rule["hard_max_age_seconds"], 345600)

    def test_real_audit_preserves_v25_seal_and_exposes_terminal_v26_recovery_failure(self):
        report = orchestrator.audit(ROOT, self.registry)
        statuses = {item["cycle_id"]: item["status"] for item in report["cycles"]}
        self.assertEqual(statuses["V25"], "SEALED_SYSTEM_PASS_PROFIT_UNPROVEN")
        self.assertEqual(statuses["V26"], "FAILED_AUTHORIZED_RECOVERY_NO_RESULT_RERUN_FORBIDDEN")
        v26 = next(item for item in report["cycles"] if item["cycle_id"] == "V26")
        self.assertTrue(v26["recovery"]["authorization_recorded"])
        self.assertFalse(v26["recovery"]["execution_allowed"])
        self.assertFalse(v26["recovery"]["metrics_available"])
        self.assertFalse(v26["recovery"]["profit_proven"])
        self.assertTrue(v26["recovery"]["next_work_order"]["v26_may_not_be_replayed"])

    def test_authority_has_no_live_broker_credential_order_or_deploy(self):
        self.assertEqual(self.registry["authority"], orchestrator.AUTHORITY)

    def test_holdout_is_machine_closed(self):
        holdout = self.registry["cycles"][0]["evaluation_contract"]["holdout"]
        self.assertEqual(holdout["state"], "UNOPENED")
        self.assertFalse(holdout["may_execute"])

    def test_mutated_raw_cost_gate_fails_preflight(self):
        cycle = copy.deepcopy(self.registry["cycles"][0])
        cycle["signal_contract"]["raw_cost_gate"] = True
        with self.assertRaisesRegex(orchestrator.ContractError, "RAW signal"):
            orchestrator.validate_cycle_contract(ROOT, cycle)

    def test_missing_one_variable_fails_preflight(self):
        cycle = copy.deepcopy(self.registry["cycles"][0])
        cycle["hypothesis_contract"]["changed_variable_count"] = 2
        with self.assertRaisesRegex(orchestrator.ContractError, "one changed variable"):
            orchestrator.validate_cycle_contract(ROOT, cycle)

    def test_legacy_registry_hash_is_immutable(self):
        self.assertEqual(
            orchestrator.sha256_file(ROOT / "PAPER_RESEARCH_CYCLE_REGISTRY_V1.json"),
            self.registry["legacy_evidence"]["registry_sha256"],
        )

    def test_coordinator_parser_preserves_nonzero_nanosecond_order_and_elapsed_time(self):
        left = orchestrator.parse_time("2026-05-01T00:00:00.123456001Z")
        right = orchestrator.parse_time("2026-05-01T00:00:00.123456789Z")
        self.assertLess(left, right)
        self.assertEqual((right - left).value, 788)
        self.assertEqual((right - left).total_seconds(), 0.000000788)


class SourceAcceptanceTest(unittest.TestCase):
    def make_source(self, complete: bool = True, duplicate_last: bool = False) -> tuple[Path, dict]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pair_dir = root / "EUR_USD"
        pair_dir.mkdir()
        source = pair_dir / "EUR_USD_M5_BA_fixture.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            for index in range(100):
                stamp_index = 98 if duplicate_last and index == 99 else index
                raw = {
                    "pair": "EUR_USD", "time": f"2026-05-01T00:{stamp_index:02d}:00.000000000Z",
                    "complete": complete, "price": "BA", "volume": 1,
                    "bid": {key: "1.0" for key in "ohlc"},
                    "ask": {key: "1.1" for key in "ohlc"},
                }
                handle.write(json.dumps(raw) + "\n")
        files = {"EUR_USD": orchestrator.sha256_file(source)}
        cycle = {"source_contract": {"root": str(root), "files": files}}
        return root, cycle

    def test_completed_bidask_source_chronology_passes(self):
        _, cycle = self.make_source()
        audit = orchestrator.validate_source(cycle)
        self.assertEqual(audit["EUR_USD"]["rows"], 100)

    def test_incomplete_source_fails_closed(self):
        _, cycle = self.make_source(complete=False)
        with self.assertRaisesRegex(orchestrator.ContractError, "non-completed"):
            orchestrator.validate_source(cycle)

    def test_nonincreasing_source_fails_closed(self):
        _, cycle = self.make_source(duplicate_last=True)
        with self.assertRaisesRegex(orchestrator.ContractError, "non-increasing"):
            orchestrator.validate_source(cycle)


class ResultAndRestartAcceptanceTest(unittest.TestCase):
    def setup_root(self) -> tuple[tempfile.TemporaryDirectory, Path, dict, dict]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "PAPER_RESEARCH_CYCLE_REGISTRY_V2.json").write_text("{}\n", encoding="utf-8")
        cycle = synthetic_cycle()
        registry = {
            "cycles": [cycle],
            "profit_gate": {
                "initial_equity_jpy": 200000,
                "normal_min_multiple": 2.0,
                "adverse_min_multiple": 2.0,
                "stretch_multiple": 3.0,
            },
        }
        return temporary, root, cycle, registry

    def test_result_validates_same_signal_cost_arms_inventory_and_terminal(self):
        temporary, root, cycle, _ = self.setup_root()
        self.addCleanup(temporary.cleanup)
        write_synthetic_result(root, cycle)
        verified = orchestrator.validate_result(root, cycle)
        self.assertEqual(verified["signals"], 1)

    def test_arm_signal_count_mismatch_fails_closed(self):
        temporary, root, cycle, _ = self.setup_root()
        self.addCleanup(temporary.cleanup)
        write_synthetic_result(root, cycle)
        path = root / cycle["execution"]["result"]
        payload = json.loads(path.read_text())
        payload["periods"]["WALK_FORWARD"]["ADVERSE_STRESS"]["source_signals"] = 0
        payload["result_sha256"] = orchestrator.embedded_hash(payload, "result_sha256")
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(orchestrator.ContractError, "signal set/count mismatch"):
            orchestrator.validate_result(root, cycle)

    def test_terminal_inventory_fails_closed(self):
        temporary, root, cycle, _ = self.setup_root()
        self.addCleanup(temporary.cleanup)
        write_synthetic_result(root, cycle)
        path = root / cycle["execution"]["result"]
        payload = json.loads(path.read_text())
        payload["periods"]["MONTH_2026_06"]["EXECUTABLE_BASE"]["terminal_open_inventory"] = 1
        payload["result_sha256"] = orchestrator.embedded_hash(payload, "result_sha256")
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(orchestrator.ContractError, "terminal inventory"):
            orchestrator.validate_result(root, cycle)

    def test_system_pass_is_separate_from_profit_and_unopened_holdout(self):
        temporary, root, cycle, registry = self.setup_root()
        self.addCleanup(temporary.cleanup)
        write_synthetic_result(root, cycle, equity=3.1)
        gates = orchestrator.evaluate_gates(registry, cycle, orchestrator.validate_result(root, cycle))
        self.assertTrue(gates["system_acceptance"]["passed"])
        self.assertFalse(gates["strategy_profit_gate"]["passed"])
        self.assertFalse(gates["strategy_profit_gate"]["unopened_holdout_reproduced"])

    def test_official_execution_is_sealed_once_and_second_run_fails(self):
        temporary, root, cycle, registry = self.setup_root()
        self.addCleanup(temporary.cleanup)

        def fake_run(*_args, **_kwargs):
            write_synthetic_result(root, cycle)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch.object(orchestrator, "validate_source", return_value={}), \
                mock.patch.object(orchestrator.subprocess, "run", side_effect=fake_run) as run:
            seal = orchestrator.execute_next(root, registry)
            self.assertEqual(seal["official_execution_ordinal"], 1)
            self.assertEqual(run.call_count, 1)
            with self.assertRaisesRegex(orchestrator.ContractError, "already has"):
                orchestrator.execute_next(root, registry)
            self.assertEqual(run.call_count, 1)

    def test_started_result_recovers_without_subprocess_rerun(self):
        temporary, root, cycle, registry = self.setup_root()
        self.addCleanup(temporary.cleanup)
        write_synthetic_result(root, cycle)
        paths = orchestrator.state_paths(root)
        orchestrator.atomic_json(paths["state"], {
            "schema_version": 2,
            "cycles": {"V25": {"status": "ATTEMPT_STARTED", "official_attempts": 1}},
        })
        with mock.patch.object(orchestrator.subprocess, "run") as run:
            seal = orchestrator.execute_next(root, registry)
        self.assertTrue(seal["recovered_without_rerun"])
        run.assert_not_called()

    def test_execute_next_skips_sealed_parent_and_runs_only_pending_child(self):
        temporary, root, parent, registry = self.setup_root()
        self.addCleanup(temporary.cleanup)
        child = copy.deepcopy(parent)
        child["cycle_id"] = "V26"
        child["depends_on_cycle"] = "V25"
        child["execution"]["result"] = "evidence/run-v26/result.json"
        child["execution"]["ledger"] = "evidence/run-v26/ledger.jsonl"
        registry["cycles"] = [parent, child]
        paths = orchestrator.state_paths(root)
        orchestrator.atomic_json(paths["state"], {
            "schema_version": 2,
            "cycles": {"V25": {"status": "SEALED_SYSTEM_PASS_PROFIT_UNPROVEN"}},
        })

        def fake_run(*_args, **_kwargs):
            write_synthetic_result(root, child)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch.object(orchestrator, "validate_source", return_value={}), \
                mock.patch.object(orchestrator.subprocess, "run", side_effect=fake_run) as run:
            seal = orchestrator.execute_next(root, registry)
        self.assertEqual(seal["cycle_id"], "V26")
        self.assertEqual(run.call_count, 1)
        self.assertTrue(orchestrator.state_paths(root, "V26")["seal"].is_file())

    def test_execute_next_skips_terminal_failed_cycle_and_runs_new_cycle(self):
        temporary, root, parent, registry = self.setup_root()
        self.addCleanup(temporary.cleanup)
        failed = copy.deepcopy(parent)
        failed["cycle_id"] = "V26"
        successor = copy.deepcopy(parent)
        successor["cycle_id"] = "V27"
        successor["depends_on_cycle"] = "V25"
        successor["execution"]["result"] = "evidence/run-v27/result.json"
        successor["execution"]["ledger"] = "evidence/run-v27/ledger.jsonl"
        registry["cycles"] = [parent, failed, successor]
        orchestrator.atomic_json(orchestrator.state_paths(root)["state"], {
            "schema_version": 2,
            "cycles": {
                "V25": {"status": "SEALED_SYSTEM_PASS_PROFIT_UNPROVEN"},
                "V26": {"status": "FAILED_AUTHORIZED_RECOVERY_NO_RERUN"},
            },
        })

        def fake_run(*_args, **_kwargs):
            write_synthetic_result(root, successor)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch.object(orchestrator, "validate_source", return_value={}), \
                mock.patch.object(orchestrator.subprocess, "run", side_effect=fake_run) as run:
            seal = orchestrator.execute_next(root, registry)
        self.assertEqual(seal["cycle_id"], "V27")
        self.assertEqual(run.call_count, 1)

    def test_failed_official_subprocess_is_terminal_and_rerun_is_forbidden(self):
        temporary, root, cycle, registry = self.setup_root()
        self.addCleanup(temporary.cleanup)
        with mock.patch.object(orchestrator, "validate_source", return_value={}), \
                mock.patch.object(orchestrator.subprocess, "run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="fixture failure"
                )) as run:
            with self.assertRaisesRegex(orchestrator.ContractError, "rerun forbidden"):
                orchestrator.execute_next(root, registry)
            with self.assertRaisesRegex(orchestrator.ContractError, "every registered cycle"):
                orchestrator.execute_next(root, registry)
        self.assertEqual(run.call_count, 1)
        state = json.loads(orchestrator.state_paths(root)["state"].read_text())
        self.assertEqual(state["cycles"]["V25"]["status"], "FAILED_OFFICIAL_EXECUTION_NO_RERUN")

    def recovery_fixture(self) -> tuple[tempfile.TemporaryDirectory, Path, dict, dict]:
        temporary, root, cycle, registry = self.setup_root()
        cycle["cycle_id"] = "V26"
        state = {
            "schema_version": 2,
            "cycles": {"V26": {
                "status": "FAILED_OFFICIAL_EXECUTION_NO_RERUN",
                "official_attempts": 1,
                "stdout_sha256": "out",
                "stderr_sha256": "err",
            }},
        }
        orchestrator.atomic_json(orchestrator.state_paths(root)["state"], state)
        return temporary, root, cycle, registry

    def test_authorized_recovery_executes_once_and_seals(self):
        temporary, root, cycle, registry = self.recovery_fixture()
        self.addCleanup(temporary.cleanup)

        def fake_run(*_args, **_kwargs):
            write_synthetic_result(root, cycle)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        authorization = {
            "authorization_sha256": orchestrator.V26_RECOVERY_AUTHORIZATION_SHA256,
        }
        with mock.patch.object(orchestrator, "validate_source", return_value={}), \
                mock.patch.object(orchestrator, "validate_v26_recovery_work_order"), \
                mock.patch.object(orchestrator, "validate_v26_recovery_authorization",
                                  return_value=authorization), \
                mock.patch.object(orchestrator.subprocess, "run", side_effect=fake_run) as run:
            seal = orchestrator.execute_v26_recovery(root, registry)
            self.assertTrue(seal["authorized_recovery_execution"])
            self.assertEqual(seal["authorized_recovery_ordinal"], 1)
            self.assertFalse(seal["recovered_without_rerun"])
            with self.assertRaisesRegex(orchestrator.ContractError, "already has"):
                orchestrator.execute_v26_recovery(root, registry)
        self.assertEqual(run.call_count, 1)

    def test_authorized_recovery_started_result_seals_without_second_subprocess(self):
        temporary, root, cycle, registry = self.recovery_fixture()
        self.addCleanup(temporary.cleanup)
        write_synthetic_result(root, cycle)
        state = json.loads(orchestrator.state_paths(root)["state"].read_text())
        state["cycles"]["V26"].update({
            "status": "RECOVERY_ATTEMPT_STARTED",
            "recovery_attempts": 1,
            "recovery_authorization_sha256": orchestrator.V26_RECOVERY_AUTHORIZATION_SHA256,
        })
        orchestrator.atomic_json(orchestrator.state_paths(root)["state"], state)
        with mock.patch.object(orchestrator.subprocess, "run") as run:
            seal = orchestrator.execute_v26_recovery(root, registry)
        self.assertTrue(seal["recovered_without_rerun"])
        self.assertTrue(seal["authorized_recovery_execution"])
        run.assert_not_called()

    def test_failed_authorized_recovery_is_terminal(self):
        temporary, root, _cycle, registry = self.recovery_fixture()
        self.addCleanup(temporary.cleanup)
        authorization = {
            "authorization_sha256": orchestrator.V26_RECOVERY_AUTHORIZATION_SHA256,
        }
        with mock.patch.object(orchestrator, "validate_source", return_value={}), \
                mock.patch.object(orchestrator, "validate_v26_recovery_work_order"), \
                mock.patch.object(orchestrator, "validate_v26_recovery_authorization",
                                  return_value=authorization), \
                mock.patch.object(orchestrator.subprocess, "run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="recovery fixture failure"
                )) as run:
            with self.assertRaisesRegex(orchestrator.ContractError, "rerun forbidden"):
                orchestrator.execute_v26_recovery(root, registry)
            with self.assertRaisesRegex(orchestrator.ContractError, "already failed"):
                orchestrator.execute_v26_recovery(root, registry)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

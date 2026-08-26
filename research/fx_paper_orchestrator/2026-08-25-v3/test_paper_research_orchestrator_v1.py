import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from paper_research_orchestrator_v1 import (
    ContractError,
    audit_cycle,
    embedded_result_hash,
    execute_next,
    load_registry,
    reconcile,
)


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "PAPER_RESEARCH_CYCLE_REGISTRY_V1.json"


class PaperResearchOrchestratorTest(unittest.TestCase):
    def test_real_registry_audits_all_cycles_without_profit_claim(self):
        registry = load_registry(ROOT, REGISTRY)
        audited = [audit_cycle(ROOT, cycle) for cycle in registry["cycles"]]
        self.assertGreaterEqual(len(audited), 4)
        self.assertTrue(all(item["status"] in {"REJECTED_DEVELOPMENT", "PENDING"} for item in audited))
        self.assertTrue(all(item["status"] == "REJECTED_DEVELOPMENT" for item in audited[:4]))

    def test_result_seal_excludes_only_its_own_hash(self):
        payload = {"live_authority": False, "external_orders": 0, "value": 7}
        seal = embedded_result_hash(payload)
        payload["result_sha256"] = seal
        self.assertEqual(embedded_result_hash(payload), seal)
        payload["value"] = 8
        self.assertNotEqual(embedded_result_hash(payload), seal)

    def test_authority_change_fails_closed(self):
        registry = json.loads(REGISTRY.read_text())
        registry["authority"]["live_authority"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "registry.json"
            path.write_text(json.dumps(registry))
            with self.assertRaises(ContractError):
                load_registry(ROOT, path)

    def test_reconcile_is_restart_safe_and_appends_journal(self):
        registry = load_registry(ROOT, REGISTRY)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            state_dir = Path(temporary) / "state"
            first = reconcile(ROOT, registry, state_dir)
            second = reconcile(ROOT, registry, state_dir)
            lines = (state_dir / "journal.jsonl").read_text().splitlines()
            current = json.loads((state_dir / "current_state.json").read_text())
            self.assertEqual(len(lines), 2)
            self.assertEqual(current["snapshot_sha256"], second["snapshot_sha256"])
            self.assertIn(first["system_status"], {"READY_FOR_NEXT_PREREGISTRATION", "READY_TO_EXECUTE_PENDING"})
            self.assertFalse(second["profit_proven"])
            self.assertIn(second["next_work_order"]["reason_code"], {
                "POSITIVE_RAW_EDGE_COST_DOMINANT", "POSITIVE_NORMAL_EDGE_ADVERSE_COST_DOMINANT",
                "PROPOSAL_RAW_EDGE_ABSENT_DESPITE_PORTFOLIO_PATH",
                "PENDING_REGISTERED_CYCLE"
            })

    def test_execute_next_pins_first_result_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            prereg = root / "prereg.json"
            prereg.write_text('{"status":"FROZEN"}\n')
            script = root / "cycle.py"
            script.write_text(
                "import hashlib,json\n"
                "from pathlib import Path\n"
                "p={'development_admitted':False,'external_orders':0,'live_authority':False,'terminal_inventory_mtm_hidden':False,'terminal_open_inventory':0}\n"
                "p['result_sha256']=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()\n"
                "q=Path('evidence/run/result.json');q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(p,sort_keys=True)+'\\n')\n"
            )
            registry_payload = {
                "schema_version": 1,
                "registry_id": "fixture",
                "authority": {
                    "paper_only": True, "live_authority": False,
                    "broker_account_access": False, "credential_access": False,
                    "order_endpoint": False, "external_orders": 0,
                    "commit_push_deploy": False,
                },
                "cycles": [{
                    "cycle_id": "C1", "depends_on": [],
                    "preregistration": "prereg.json",
                    "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
                    "script": "cycle.py", "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                    "result": "evidence/run/result.json",
                    "admission_field": "development_admitted", "admission_operator": "IS_TRUE",
                    "execution_class": "LOCAL_REPLAY_ONLY",
                    "arguments": ["--output-root", "evidence/run"],
                }],
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry_payload))
            registry = load_registry(root, registry_path)
            state = execute_next(root, registry, root / "state")
            self.assertEqual(state["cycles"][0]["status"], "REJECTED_DEVELOPMENT")
            result = root / "evidence/run/result.json"
            result.write_text(result.read_text() + " ")
            with self.assertRaises(ContractError):
                reconcile(root, registry, root / "state")


if __name__ == "__main__":
    unittest.main()

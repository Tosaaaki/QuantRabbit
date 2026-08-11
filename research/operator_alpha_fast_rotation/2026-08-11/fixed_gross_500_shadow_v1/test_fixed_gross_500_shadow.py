from __future__ import annotations
import importlib.util
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
S=importlib.util.spec_from_file_location("shadow",ROOT/"run_fixed_gross_500_shadow.py");M=importlib.util.module_from_spec(S);assert S.loader;S.loader.exec_module(M)
class FixedGrossShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.report=M.run()
    def test_target_price_is_executable_gross_math(self):
        self.assertAlmostEqual(M.target_price("LONG",100.0,1000,1.0,2.0,{"kind":"JPY","value":500.0}),100.5)
        self.assertAlmostEqual(M.pnl("LONG",100.0,100.5,1000,1.0),500.0)
    def test_no_end_of_replay_forced_loss_close(self):
        rows=[json.loads(x) for x in (ROOT/"decision_results_v1.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["status"]!="FORCED_END_CLOSE" for r in rows))
        self.assertTrue(all(r["reentry_eligible"] is False for r in rows if r["status"]=="UNRESOLVED_MTM"))
    def test_new_hedge_context_is_not_counted_as_clean_concurrency_one_success(self):
        self.assertEqual(self.report["new_manual_trade_classification"]["classification"],"OVERLAPPING_HEDGE_CONTEXT_NOT_CONCURRENCY_ONE_SUCCESS")
    def test_all_four_arms_use_identical_frozen_decision_ids(self):
        rows=[json.loads(x) for x in (ROOT/"decision_results_v1.jsonl").read_text().splitlines()]
        by={}
        for r in rows: by.setdefault(r["arm"],set()).add(r["decision_id"])
        self.assertEqual(len({tuple(sorted(v)) for v in by.values()}),1)
    def test_live_permission_never_appears(self): self.assertFalse(self.report["permissions"]["live"])
    def test_four_prior_manual_wins_reach_fixed_target(self):
        rows=[json.loads(x) for x in (ROOT/"decision_results_v1.jsonl").read_text().splitlines()]
        ids={"473189->473191","473193->473195","473197->473199","473201->473204"}
        self.assertTrue(all(r["status"]=="TAKE_PROFIT" for r in rows if r["arm"]=="FIXED_GROSS_500" and r["cohort_id"] in ids))
if __name__=="__main__": unittest.main()

from __future__ import annotations
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class OOSReplayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.v1=json.loads((ROOT/"summary_all.json").read_text());cls.v2=json.loads((ROOT/"confirmatory_v2/summary_all.json").read_text());cls.oracle=json.loads((ROOT/"independent_oracle_v1.json").read_text())
 def test_standard_replay_invariants(self):
  for s in (self.v1,self.v2):
   self.assertTrue(s["standard_invariants"]["hard_tp_enabled"]);self.assertFalse(s["standard_invariants"]["hard_sl_enabled"]);self.assertTrue(s["standard_invariants"]["end_of_replay_forced_close_excluded"])
 def test_same_frozen_candidate_count_for_all_arms(self):
  for s in (self.v1,self.v2):self.assertEqual(len({x["candidate_signals"] for x in s["arms"].values()}),1)
 def test_unresolved_mtm_is_in_terminal_equity(self):
  for s in (self.v1,self.v2):
   for v in s["arms"].values():
    self.assertGreaterEqual(v["unresolved_inventory"],1);self.assertLess(v["terminal_equity_jpy"],254209.0185)
 def test_entry_only_iteration_did_not_promote(self):
  self.assertEqual(self.v2["adoption"],"HOLD");self.assertTrue(all(not x["positive_terminal_expectancy"] for x in self.v2["adoption_gates"].values()))
 def test_oracle(self):self.assertTrue(self.oracle["checks"]["all_accounting_checks"]);self.assertTrue(self.oracle["checks"]["v1_causality"]);self.assertTrue(self.oracle["checks"]["v2_causality"])
 def test_fixed_target_is_least_bad_confirmatory_terminal(self):
  eq={a:v["terminal_equity_jpy"] for a,v in self.v2["arms"].items()};self.assertEqual(max(eq,key=eq.get),"FIXED_GROSS_500")
if __name__=="__main__":unittest.main()

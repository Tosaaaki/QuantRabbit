#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;START=254209.0185
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(summary_path,receipt_path):
 s=json.loads(summary_path.read_text());rows=[json.loads(x) for x in receipt_path.read_text().splitlines() if x];out={}
 for arm,v in s["arms"].items():
  r=[x for x in rows if x["arm"]==arm];completed=[x for x in r if x["status"]=="TAKE_PROFIT"];unresolved=[x for x in r if x["status"]=="UNRESOLVED_MTM"];cash=sum(float(x["after_cost_jpy"]) for x in r);mtm=sum(float(x["terminal_mtm_jpy"]) for x in unresolved);terminal=START+cash+mtm
  out[arm]={"executed_match":len(r)==v["executed_cycles"],"completed_match":len(completed)==v["completed_cycles"],"unresolved_match":len(unresolved)==v["unresolved_inventory"],"terminal_equity_match":abs(terminal-v["terminal_equity_jpy"])<1e-6,"forced_end_close_absent":all(x["status"]!="FORCED_END_CLOSE" for x in r),"oracle_terminal_equity_jpy":terminal}
 return out
def causality(path):
 for line in path.read_text().splitlines():
  r=json.loads(line)
  if not (r["h4_complete_through_utc"]<=r["decision_utc"] and r["m1_complete_through_utc"]<=r["entry_utc"] and r["s5_complete_through_utc"]<=r["entry_utc"]):return False
 return True
def main():
 v1=verify(ROOT/"summary_all.json",ROOT/"decision_receipts_v1.jsonl");v2=verify(ROOT/"confirmatory_v2/summary_all.json",ROOT/"confirmatory_v2/decision_receipts_v2.jsonl")
 m1=json.loads((ROOT/"signal_manifest_v1.json").read_text());m2=json.loads((ROOT/"confirmatory_v2/signal_manifest_v2.json").read_text())
 out={"contract":"OPERATOR_ALPHA_OOS_INDEPENDENT_ORACLE_V1","diagnostic_v1":v1,"confirmatory_v2":v2,"checks":{"v1_signal_hash_match":sha(ROOT/"frozen_signal_log_v1.jsonl")==m1["signal_log"]["sha256"],"v2_signal_hash_match":sha(ROOT/"confirmatory_v2/frozen_signal_log_v2.jsonl")==m2["signal_log"]["sha256"],"v1_causality":causality(ROOT/"frozen_signal_log_v1.jsonl"),"v2_causality":causality(ROOT/"confirmatory_v2/frozen_signal_log_v2.jsonl"),"all_accounting_checks":all(all(x.values()) for phase in (v1,v2) for x in phase.values())}}
 (ROOT/"independent_oracle_v1.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

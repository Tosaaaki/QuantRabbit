#!/usr/bin/env python3
"""Independent arithmetic/readback check; intentionally does not import the replay module."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main() -> int:
    rows=[json.loads(x) for x in (ROOT/"decision_results_v1.jsonl").read_text().splitlines() if x]
    out={"contract":"FIXED_GROSS_500_INDEPENDENT_ORACLE_V1","arms":{},"checks":{}}
    for arm in sorted({r["arm"] for r in rows}):
        v=[r for r in rows if r["arm"]==arm]; total=sum(float(r["terminal_contribution_jpy"]) for r in v if r["terminal_contribution_jpy"] is not None)
        out["arms"][arm]={"decision_ids":sorted(r["decision_id"] for r in v),"terminal_sum_jpy":total,"take_profit_count":sum(r["status"]=="TAKE_PROFIT" for r in v),"unresolved_count":sum(r["status"]=="UNRESOLVED_MTM" for r in v)}
    fixed=[r for r in rows if r["arm"]=="FIXED_GROSS_500"]
    clean=[r for r in fixed if r["cohort_id"] in {"473189->473191","473193->473195","473197->473199","473201->473204"}]
    out["checks"]={"all_arms_same_decision_ids":len({tuple(x["decision_ids"]) for x in out["arms"].values()})==1,"no_forced_end_close":all(r["status"]!="FORCED_END_CLOSE" for r in rows),"four_prior_manual_wins_hit_fixed_500":all(r["status"]=="TAKE_PROFIT" for r in clean),"four_prior_manual_mean_seconds":sum(r["holding_seconds"] for r in clean)/len(clean),"new_560_not_clean_concurrency_one":next(r for r in fixed if r["cohort_id"]=="473212->473218")["status"]=="UNRESOLVED_MTM"}
    (ROOT/"independent_oracle_v1.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

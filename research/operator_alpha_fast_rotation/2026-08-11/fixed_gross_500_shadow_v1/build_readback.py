#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main() -> int:
 r=json.loads((ROOT/"comparison_v1.json").read_text());o=json.loads((ROOT/"independent_oracle_v1.json").read_text());m=json.loads((ROOT/"source_manifest_v1.json").read_text())
 lines=["# Fixed gross +500 shadow readback","","- Environment: research/shadow only; no live/Paper/order/deploy write.","- Adoption: HOLD.","- Fresh broker truth: transaction 473212→473218 realized +560 JPY, but it overlapped still-open 473207; excluded from the clean concurrency=1 success denominator.","- Flat check: one non-shadow/manual-or-unknown trade remained open, so no re-entry is represented as broker-confirmed for that account snapshot.","", "| Arm | TP / 7 | Unresolved | After-cost terminal JPY | Mean hold sec |", "|---|---:|---:|---:|---:|"]
 for k,v in r["arms"].items(): lines.append(f"| {k} | {v['take_profit']} | {v['unresolved']} | {v['after_cost_terminal_jpy']:.2f} | {v['mean_holding_seconds']:.1f} |")
 c=o['checks'];lines += ["",f"- Independent oracle: same decision IDs={c['all_arms_same_decision_ids']}; forced end close absent={c['no_forced_end_close']}; prior four clean manual wins reached fixed +500={c['four_prior_manual_wins_hit_fixed_500']}; mean target-touch={c['four_prior_manual_mean_seconds']:.1f}s.","- Signal supply: contract is present but remains NOT_EVALUABLE until a frozen out-of-sample S5/M1 signal log exists; it does not extrapolate 100 trades/day or returns.","- Standard replay worker `scripts/replay_exit_workers_groups.py` is absent in this revision, so this uses the research-local S5 bid/ask replay and is not a substitute for that standard worker."]
 (ROOT/"readback_v1.md").write_text("\n".join(lines)+"\n")
 print("\n".join(lines));return 0
if __name__=="__main__":raise SystemExit(main())

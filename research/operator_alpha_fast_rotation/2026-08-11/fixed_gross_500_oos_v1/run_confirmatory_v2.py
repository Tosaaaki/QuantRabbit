#!/usr/bin/env python3
"""Confirmatory entry-only iteration on a later untouched period."""
from __future__ import annotations
import bisect
import importlib.util
import json
import math
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from pathlib import Path

BASE=Path(__file__).resolve().parent;ROOT=BASE/"confirmatory_v2"
spec=importlib.util.spec_from_file_location("oos_v1",BASE/"run_oos_signal_replay.py");M=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(M)
M.ROOT=ROOT;M.START=datetime(2025,1,13,tzinfo=timezone.utc);M.END=datetime(2025,4,1,tzinfo=timezone.utc);M.WARM=datetime(2025,1,1,tzinfo=timezone.utc)
S5ROOT=Path("/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_s5_2024_2026/split_by_year")
M.S5={p:S5ROOT/p/f"{p}_S5_BA_20250101T000000Z_20260101T000000Z.jsonl.gz" for p in M.PAIRS}
M1ROOT=Path("/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026")
M.M1={"EUR_USD":M1ROOT/"20260718T085350Z/EUR_USD/EUR_USD_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz","USD_JPY":M1ROOT/"20260718T084558Z/USD_JPY/USD_JPY_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz","GBP_USD":M1ROOT/"20260718T104309Z/GBP_USD/GBP_USD_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz","AUD_USD":M1ROOT/"20260718T105320Z/AUD_USD/AUD_USD_M1_BA_20250101T000000Z_20260101T000000Z.jsonl.gz"}
DEPTHS=(0.25,0.10)

def daily_dirs(m1):
 buckets=[];cur=None;last=None
 for r in m1:
  b=r[0].replace(hour=0,minute=0,second=0,microsecond=0)
  if cur is not None and b!=cur:buckets.append((cur+timedelta(days=1),last))
  cur=b;last=M.mid(r)
 if cur is not None:buckets.append((cur+timedelta(days=1),last))
 xs=[x[1] for x in buckets];e5=M.ema_series(xs,5);e10=M.ema_series(xs,10);dirs=[]
 for i,x in enumerate(xs):dirs.append("LONG" if i>=9 and e5[i]>e10[i] and x>e10[i] else "SHORT" if i>=9 and e5[i]<e10[i] and x<e10[i] else "NEUTRAL")
 return [x[0] for x in buckets],dirs
def build_pair(pair):
 m1=M.load_bars(M.M1[pair],M.WARM,M.END);s5=M.load_bars(M.S5[pair],M.START-timedelta(minutes=5),M.END);s5t=[r[0] for r in s5];h4t,h4d=M.h4_directions(m1);d1t,d1d=daily_dirs(m1);mids=[M.mid(r) for r in m1];e8=M.ema_series(mids,8);out={d:[] for d in DEPTHS};used={d:set() for d in DEPTHS}
 for i in range(22,len(m1)-1):
  decision=m1[i][0]+timedelta(minutes=1)
  if decision<M.START or decision>=M.END or decision.hour>=22:continue
  h=bisect.bisect_right(h4t,decision)-1;d=bisect.bisect_right(d1t,decision)-1
  if h<0 or d<0 or h4d[h]=="NEUTRAL" or h4d[h]!=d1d[d]:continue
  side=h4d[h];k=bisect.bisect_left(s5t,decision)
  if k<3 or k>=len(s5):continue
  recent=s5[k-3:k]
  if not all(M.directional(side,M.mid(x,1),M.mid(x,4)) for x in recent):continue
  reclaim=(mids[i-1]<=e8[i-1] and mids[i]>e8[i]) if side=="LONG" else (mids[i-1]>=e8[i-1] and mids[i]<e8[i])
  if not reclaim:continue
  atr=M.atr14(m1,i);look=mids[max(0,i-15):i+1];adverse=(max(look)-min(mids[i-1],mids[i])) if side=="LONG" else (max(mids[i-1],mids[i])-min(look));bucket=(pair,h4t[h])
  for depth in DEPTHS:
   if adverse<depth*atr or bucket in used[depth]:continue
   entry=s5[k];price=entry[5] if side=="LONG" else entry[1];name=f"D1_H4_DEEP_{int(depth*100):03d}"
   out[depth].append({"decision_id":f"OOS2-{name}-{pair}-{decision.strftime('%Y%m%dT%H%M%SZ')}","variant":name,"pair":pair,"side":side,"decision_utc":M.iso(decision),"entry_utc":M.iso(entry[0]),"entry_index":k,"entry_price":price,"atr_m1":atr,"h4_complete_through_utc":M.iso(h4t[h]),"d1_complete_through_utc":M.iso(d1t[d]),"m1_complete_through_utc":M.iso(decision),"s5_complete_through_utc":M.iso(s5[k-1][0]+timedelta(seconds=5)),"entry_probability":1.0,"entry_units_intent":M.UNITS,"pullback_depth_atr":depth});used[depth].add(bucket)
 return out,{"m1_rows":len(m1),"s5_rows":len(s5)}
def build():
 allv={d:[] for d in DEPTHS};coverage={}
 for p in M.PAIRS:
  rows,c=build_pair(p);coverage[p]=c
  for d in DEPTHS:allv[d]+=rows[d]
 supply={d:{"raw_signals":len(allv[d]),"observed_signal_days":len({M.ts(x["entry_utc"]).date() for x in allv[d]}),"signals_per_observed_day":len(allv[d])/max(1,len({M.ts(x["entry_utc"]).date() for x in allv[d]}))} for d in DEPTHS};selected=next((d for d in DEPTHS if supply[d]["raw_signals"]>=150 and supply[d]["signals_per_observed_day"]>=2),DEPTHS[-1]);rows=sorted(allv[selected],key=lambda x:(x["entry_utc"],x["pair"]));path=ROOT/"frozen_signal_log_v2.jsonl";path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows));manifest={"contract":"OPERATOR_ALPHA_CONFIRMATORY_SIGNAL_MANIFEST_V2","built_before_outcome_replay":True,"selected_depth_atr":selected,"selection_uses_outcomes":False,"supply_only_iteration":{str(k):v for k,v in supply.items()},"date_range":{"start":M.iso(M.START),"end_exclusive":M.iso(M.END)},"pairs":list(M.PAIRS),"coverage":coverage,"sources":{p:{"s5_path":str(M.S5[p]),"s5_sha256":M.file_sha(M.S5[p]),"m1_path":str(M.M1[p]),"m1_sha256":M.file_sha(M.M1[p])} for p in M.PAIRS},"signal_log":{"rows":len(rows),"sha256":M.file_sha(path)}};(ROOT/"signal_manifest_v2.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"selected_depth":selected,"supply":supply,"rows":len(rows)},indent=2))
def replay():
 man=json.loads((ROOT/"signal_manifest_v2.json").read_text());path=ROOT/"frozen_signal_log_v2.jsonl"
 if M.file_sha(path)!=man["signal_log"]["sha256"]:raise RuntimeError("signal hash mismatch")
 signals=[json.loads(x) for x in path.read_text().splitlines() if x];by=defaultdict(list)
 for x in signals:by[x["pair"]].append(x)
 usd=M.load_bars(M.S5["USD_JPY"],M.START,M.END);lookup=([x[0] for x in usd],usd);out={}
 for p in M.PAIRS:out.update(M.arm_outcomes(p,by[p],usd if p=="USD_JPY" else M.load_bars(M.S5[p],M.START,M.END),lookup))
 summaries={};receipts=[]
 for arm in M.ARMS:summaries[arm],r=M.summarize(arm,signals,out);receipts+=r
 gates={a:{"minimum_completed_cycles":v["completed_cycles"]>=100,"positive_terminal_expectancy":v["terminal_expectancy_per_execution_jpy"]>0,"unresolved_zero":v["unresolved_inventory"]==0,"margin_closeouts_zero":v["margin_closeouts"]==0} for a,v in summaries.items()};result={"contract":"OPERATOR_ALPHA_CONFIRMATORY_SUMMARY_ALL_V2","canonical_summary":True,"standard_invariants":{"hard_tp_enabled":True,"hard_sl_enabled":False,"end_of_replay_forced_close_excluded":True},"selected_depth_atr":man["selected_depth_atr"],"arms":summaries,"adoption_gates":gates,"adoption":"HOLD"};(ROOT/"decision_receipts_v2.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in receipts));(ROOT/"summary_all.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":
 import argparse;p=argparse.ArgumentParser();p.add_argument("--build",action="store_true");p.add_argument("--replay",action="store_true");a=p.parse_args();build() if a.build else replay() if a.replay else p.error("choose --build or --replay")

#!/usr/bin/env python3
"""Freeze and replay an OOS operator-alpha S5/M1 signal cohort.

The local runner preserves qr-replay-backtest invariants: hard TP stays on,
hard SL is absent, and end-of-replay inventory is marked rather than closed.
It has no broker or order client.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT=Path(__file__).resolve().parent
START=datetime(2024,10,1,tzinfo=timezone.utc);END=datetime(2025,1,1,tzinfo=timezone.utc);WARM=datetime(2024,9,1,tzinfo=timezone.utc)
PAIRS=("EUR_USD","USD_JPY","GBP_USD","AUD_USD");UNITS=5000;START_EQUITY=254209.0185
S5_ROOT=Path("/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_s5_2024_2026/split_by_year")
M1_ROOT=Path("/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026")
S5={p:S5_ROOT/p/f"{p}_S5_BA_20240101T000000Z_20250101T000000Z.jsonl.gz" for p in PAIRS}
M1={"EUR_USD":M1_ROOT/"20260718T085236Z/EUR_USD/EUR_USD_M1_BA_20240101T000000Z_20250101T000000Z.jsonl.gz","USD_JPY":M1_ROOT/"20260718T084445Z/USD_JPY/USD_JPY_M1_BA_20240101T000000Z_20250101T000000Z.jsonl.gz","GBP_USD":M1_ROOT/"20260718T104116Z/GBP_USD/GBP_USD_M1_BA_20240101T000000Z_20250101T000000Z.jsonl.gz","AUD_USD":M1_ROOT/"20260718T105139Z/AUD_USD/AUD_USD_M1_BA_20240101T000000Z_20250101T000000Z.jsonl.gz"}
ARMS={"FIXED_GROSS_500":("JPY",500.0),"ATR_010":("ATR",0.10),"ATR_020":("ATR",0.20),"ATR_025":("ATR",0.25)}
VARIANTS=("STRICT_PULLBACK_RECLAIM","BALANCED_EMA_CONTINUATION","PERMISSIVE_DIRECTIONAL_CONTINUATION")

def ts(v:str)->datetime:
 v=v.replace("Z","+00:00");
 if "." in v:
  h,r=v.split(".",1);f,z=r.split("+",1);v=f"{h}.{f[:6]}+{z}"
 return datetime.fromisoformat(v).astimezone(timezone.utc)
def iso(v:datetime)->str:return v.astimezone(timezone.utc).isoformat().replace("+00:00","Z")
def file_sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def canonical_sha(v:Any)->str:return hashlib.sha256((json.dumps(v,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
def load_bars(path:Path,start:datetime,end:datetime)->list[tuple[Any,...]]:
 out=[]
 with gzip.open(path,"rt",encoding="utf-8") as f:
  for line in f:
   r=json.loads(line);t=ts(str(r["time"]))
   if t<start:continue
   if t>=end:break
   if r.get("complete") is not True:continue
   b,a=r.get("bid"),r.get("ask")
   if not isinstance(b,dict) or not isinstance(a,dict):continue
   vals=(t,float(b["o"]),float(b["h"]),float(b["l"]),float(b["c"]),float(a["o"]),float(a["h"]),float(a["l"]),float(a["c"]))
   if all(math.isfinite(x) for x in vals[1:]) and vals[4]<vals[8]:out.append(vals)
 return out
def mid(r:tuple[Any,...],i:int=4)->float:
 return (r[i]+r[i+4])/2
def ema_series(xs:list[float],n:int)->list[float]:
 a=2/(n+1);out=[];v=xs[0]
 for x in xs:v=a*x+(1-a)*v;out.append(v)
 return out
def h4_directions(m1:list[tuple[Any,...]])->tuple[list[datetime],list[str]]:
 buckets=[];cur=None;last=None
 for r in m1:
  b=r[0].replace(hour=(r[0].hour//4)*4,minute=0,second=0,microsecond=0)
  if cur is not None and b!=cur:buckets.append((cur+timedelta(hours=4),last))
  cur=b;last=mid(r)
 if cur is not None:buckets.append((cur+timedelta(hours=4),last))
 closes=[x[1] for x in buckets];e8=ema_series(closes,8);e21=ema_series(closes,21);dirs=[]
 for i,x in enumerate(closes):dirs.append("LONG" if i>=20 and e8[i]>e21[i] and x>e21[i] else "SHORT" if i>=20 and e8[i]<e21[i] and x<e21[i] else "NEUTRAL")
 return [x[0] for x in buckets],dirs
def atr14(m1:list[tuple[Any,...]],i:int)->float:
 vals=[]
 for j in range(i-13,i+1):
  hi=max(m1[j][2],m1[j][6]);lo=min(m1[j][3],m1[j][7]);pc=mid(m1[j-1]);vals.append(max(hi-lo,abs(hi-pc),abs(lo-pc)))
 return mean(vals)
def directional(side:str,a:float,b:float)->bool:return b>a if side=="LONG" else b<a
def build_pair_signals(pair:str)->tuple[dict[str,list[dict[str,Any]]],dict[str,Any]]:
 m1=load_bars(M1[pair],WARM,END);s5=load_bars(S5[pair],START-timedelta(minutes=5),END);s5t=[r[0] for r in s5];h4t,h4d=h4_directions(m1);mids=[mid(r) for r in m1];e8=ema_series(mids,8)
 out={v:[] for v in VARIANTS};last={v:datetime.min.replace(tzinfo=timezone.utc) for v in VARIANTS}
 for i in range(22,len(m1)-1):
  decision=m1[i][0]+timedelta(minutes=1)
  if decision<START or decision>=END or decision.hour>=22:continue
  h=bisect.bisect_right(h4t,decision)-1
  if h<0 or h4d[h]=="NEUTRAL":continue
  side=h4d[h];k=bisect.bisect_left(s5t,decision)
  if k<3 or k>=len(s5):continue
  recent=s5[k-3:k];s5_3=all(directional(side,mid(x,1),mid(x,4)) for x in recent);s5_2=all(directional(side,mid(x,1),mid(x,4)) for x in recent[-2:]);s5_1=directional(side,mid(recent[-1],1),mid(recent[-1],4))
  reclaim=(mids[i-1]<=e8[i-1] and mids[i]>e8[i]) if side=="LONG" else (mids[i-1]>=e8[i-1] and mids[i]<e8[i])
  continuation=(mids[i]>e8[i] and mids[i]>mids[i-1]) if side=="LONG" else (mids[i]<e8[i] and mids[i]<mids[i-1])
  permissive=directional(side,mids[i-1],mids[i])
  flags={VARIANTS[0]:reclaim and s5_3,VARIANTS[1]:continuation and s5_2,VARIANTS[2]:permissive and s5_1}
  for v,ok in flags.items():
   if not ok or (decision-last[v]).total_seconds()<300:continue
   entry=s5[k];price=entry[5] if side=="LONG" else entry[1]
   row={"decision_id":f"OOS-{v}-{pair}-{decision.strftime('%Y%m%dT%H%M%SZ')}","variant":v,"pair":pair,"side":side,"decision_utc":iso(decision),"entry_utc":iso(entry[0]),"entry_index":k,"entry_price":price,"atr_m1":atr14(m1,i),"h4_complete_through_utc":iso(h4t[h]),"h4_direction":side,"m1_complete_through_utc":iso(decision),"s5_complete_through_utc":iso(s5[k-1][0]+timedelta(seconds=5)),"entry_probability":1.0,"entry_units_intent":UNITS}
   out[v].append(row);last[v]=decision
 return out,{"m1_rows":len(m1),"s5_rows":len(s5),"first_m1":iso(m1[0][0]),"last_m1":iso(m1[-1][0]),"first_s5":iso(s5[0][0]),"last_s5":iso(s5[-1][0])}
def build_signals()->dict[str,Any]:
 allv={v:[] for v in VARIANTS};coverage={}
 for pair in PAIRS:
  rows,cov=build_pair_signals(pair);coverage[pair]=cov
  for v in VARIANTS:allv[v].extend(rows[v])
 observed_days=len({ts(r["entry_utc"]).date() for v in VARIANTS for r in allv[v]}) or 1
 supply={v:{"raw_signals":len(allv[v]),"observed_signal_days":len({ts(r["entry_utc"]).date() for r in allv[v]}),"signals_per_observed_day":len(allv[v])/max(1,len({ts(r["entry_utc"]).date() for r in allv[v]}))} for v in VARIANTS}
 selected=next((v for v in VARIANTS if supply[v]["raw_signals"]>=150 and supply[v]["signals_per_observed_day"]>=2.0),VARIANTS[-1])
 rows=sorted(allv[selected],key=lambda x:(x["entry_utc"],x["pair"],x["decision_id"]))
 path=ROOT/"frozen_signal_log_v1.jsonl";path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8")
 manifest={"contract":"OPERATOR_ALPHA_OOS_SIGNAL_MANIFEST_V1","built_before_outcome_replay":True,"selected_trigger":selected,"trigger_selection_uses_outcomes":False,"date_range":{"start":iso(START),"end_exclusive":iso(END)},"pairs":list(PAIRS),"price_component":"BID_ASK","complete_only":True,"supply_only_trigger_iteration":supply,"coverage":coverage,"sources":{p:{"s5_path":str(S5[p]),"s5_sha256":file_sha(S5[p]),"m1_path":str(M1[p]),"m1_sha256":file_sha(M1[p])} for p in PAIRS},"signal_log":{"rows":len(rows),"sha256":file_sha(path)}}
 (ROOT/"signal_manifest_v1.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({"selected":selected,"supply":supply,"rows":len(rows)},indent=2));return manifest
def q_to_jpy(pair:str,entry_time:datetime,usd_times:list[datetime],usd_rows:list[tuple[Any,...]])->float:
 if pair.endswith("JPY"):return 1.0
 i=bisect.bisect_right(usd_times,entry_time)-1;return mid(usd_rows[i]) if i>=0 else 150.0
def arm_outcomes(pair:str,signals:list[dict[str,Any]],s5:list[tuple[Any,...]],q_lookup:tuple[list[datetime],list[tuple[Any,...]]])->dict[str,dict[str,Any]]:
 out={};usd_times,usd_rows=q_lookup
 for sig in signals:
  i=int(sig["entry_index"]);side=sig["side"];entry=float(sig["entry_price"]);atr=float(sig["atr_m1"]);q=q_to_jpy(pair,ts(sig["entry_utc"]),usd_times,usd_rows);notional=entry*UNITS*q;margin=notional*0.04
  targets={a:(entry+(v/(UNITS*q) if k=="JPY" else atr*v)*(1 if side=="LONG" else -1)) for a,(k,v) in ARMS.items()};pending=set(ARMS);state={a:{"min_mtm":0.0,"closeout":False} for a in ARMS}
  last=s5[i];end_index=len(s5)-1
  for j in range(i+1,len(s5)):
   r=s5[j];last=r
   exit_close=r[4] if side=="LONG" else r[8];mtm=(exit_close-entry)*UNITS*q*(1 if side=="LONG" else -1)
   for a in tuple(pending):
    state[a]["min_mtm"]=min(state[a]["min_mtm"],mtm)
    if START_EQUITY+mtm<=margin*0.5:
     state[a].update({"status":"MARGIN_CLOSEOUT","exit_utc":iso(r[0]+timedelta(seconds=5)),"flat_confirm_utc":iso(r[0]+timedelta(seconds=10)),"gross_jpy":mtm,"after_cost_jpy":mtm,"terminal_mtm_jpy":0.0,"holding_seconds":((r[0]+timedelta(seconds=5))-ts(sig["entry_utc"])).total_seconds(),"closeout":True});pending.remove(a);continue
    touched=(r[2]>=targets[a]) if side=="LONG" else (r[7]<=targets[a])
    if touched:
     spread=r[8]-r[4];slip=spread*0.10*UNITS*q;gross=(targets[a]-entry)*UNITS*q*(1 if side=="LONG" else -1);days=max(0,(r[0].date()-ts(sig["entry_utc"]).date()).days);fin=-notional*0.015/365*days
     state[a].update({"status":"TAKE_PROFIT","exit_utc":iso(r[0]+timedelta(seconds=5)),"flat_confirm_utc":iso(r[0]+timedelta(seconds=10)),"gross_jpy":gross,"after_cost_jpy":gross-slip+fin,"terminal_mtm_jpy":0.0,"slippage_jpy":slip,"financing_stress_jpy":fin,"holding_seconds":((r[0]+timedelta(seconds=5))-ts(sig["entry_utc"])).total_seconds(),"target_price":targets[a]});pending.remove(a)
   if not pending:break
  for a in pending:
   px=last[4] if side=="LONG" else last[8];mtm=(px-entry)*UNITS*q*(1 if side=="LONG" else -1);days=max(0,(last[0].date()-ts(sig["entry_utc"]).date()).days);fin=-notional*0.015/365*days
   state[a].update({"status":"UNRESOLVED_MTM","exit_utc":None,"flat_confirm_utc":None,"gross_jpy":None,"after_cost_jpy":fin,"terminal_mtm_jpy":mtm,"financing_stress_jpy":fin,"holding_seconds":((last[0]+timedelta(seconds=5))-ts(sig["entry_utc"])).total_seconds(),"target_price":targets[a]})
  out[sig["decision_id"]]={a:{**state[a],"margin_required_jpy":margin,"notional_jpy":notional,"quote_to_jpy":q} for a in ARMS}
 return out
def summarize(arm:str,signals:list[dict[str,Any]],outcomes:dict[str,dict[str,Any]])->tuple[dict[str,Any],list[dict[str,Any]]]:
 occupied=START;cash=0.0;high=0.0;max_dd=0.0;receipts=[];skipped=0
 for sig in signals:
  entry=ts(sig["entry_utc"])
  if entry<occupied:skipped+=1;continue
  o=outcomes[sig["decision_id"]][arm];receipt={"decision_id":sig["decision_id"],"pair":sig["pair"],"side":sig["side"],"entry_utc":sig["entry_utc"],"arm":arm,**o};receipts.append(receipt)
  if o["flat_confirm_utc"]:occupied=ts(o["flat_confirm_utc"])
  else:occupied=END
  cash+=float(o["after_cost_jpy"]);high=max(high,cash);max_dd=max(max_dd,high-cash,-float(o["min_mtm"]))
 completed=[r for r in receipts if r["status"]=="TAKE_PROFIT"];loss=[-r["after_cost_jpy"] for r in completed if r["after_cost_jpy"]<0];wins=[r["after_cost_jpy"] for r in completed if r["after_cost_jpy"]>0];unresolved=[r for r in receipts if r["status"]=="UNRESOLVED_MTM"];terminal=START_EQUITY+cash+sum(r["terminal_mtm_jpy"] for r in unresolved)
 days=(END-START).total_seconds()/86400;by_hour=Counter(ts(r["entry_utc"]).hour for r in receipts);candidate_by_hour=Counter(ts(r["entry_utc"]).hour for r in signals);per_day=len(receipts)/days;completed_day=len(completed)/days
 calendar100=(100/completed_day) if completed_day>0 else None
 summary={"candidate_signals":len(signals),"executed_cycles":len(receipts),"skipped_while_occupied":skipped,"completed_cycles":len(completed),"clean_cycle_completion_rate":len(completed)/len(receipts) if receipts else 0.0,"gross_target_reach_rate":len(completed)/len(receipts) if receipts else 0.0,"signals_per_calendar_day":len(signals)/days,"executions_per_calendar_day":per_day,"completed_cycles_per_calendar_day":completed_day,"calendar_days_for_100_completed_cycles":calendar100,"candidate_signals_per_hour_utc":dict(sorted(candidate_by_hour.items())),"executions_per_hour_utc":dict(sorted(by_hour.items())),"mean_time_to_target_seconds":mean([r["holding_seconds"] for r in completed]) if completed else None,"median_time_to_target_seconds":median([r["holding_seconds"] for r in completed]) if completed else None,"p90_time_to_target_seconds":sorted([r["holding_seconds"] for r in completed])[max(0,math.ceil(len(completed)*0.9)-1)] if completed else None,"unresolved_inventory":len(unresolved),"terminal_unrealized_mtm_jpy":sum(r["terminal_mtm_jpy"] for r in unresolved),"after_cost_cash_delta_jpy":cash,"terminal_equity_jpy":terminal,"expectancy_per_completed_cycle_jpy":mean([r["after_cost_jpy"] for r in completed]) if completed else None,"terminal_expectancy_per_execution_jpy":((terminal-START_EQUITY)/len(receipts)) if receipts else None,"profit_factor":sum(wins)/sum(loss) if loss else None,"profit_factor_status":"FINITE" if loss else "NO_COMPLETED_LOSS","max_equity_drawdown_jpy":max_dd,"margin_closeouts":sum(r["status"]=="MARGIN_CLOSEOUT" for r in receipts),"max_margin_required_jpy":max([r["margin_required_jpy"] for r in receipts],default=0.0),"margin_occupancy_fraction":sum(min((ts(r["flat_confirm_utc"]) if r["flat_confirm_utc"] else END)-ts(r["entry_utc"]),END-ts(r["entry_utc"])).total_seconds() for r in receipts)/((END-START).total_seconds()) if receipts else 0.0}
 return summary,receipts
def replay()->dict[str,Any]:
 manifest=json.loads((ROOT/"signal_manifest_v1.json").read_text());path=ROOT/"frozen_signal_log_v1.jsonl"
 if file_sha(path)!=manifest["signal_log"]["sha256"]:raise RuntimeError("frozen signal hash mismatch")
 signals=[json.loads(x) for x in path.read_text().splitlines() if x];by_pair=defaultdict(list)
 for s in signals:by_pair[s["pair"]].append(s)
 usd=load_bars(S5["USD_JPY"],START,END);lookup=([r[0] for r in usd],usd);outcomes={}
 for pair in PAIRS:
  bars=usd if pair=="USD_JPY" else load_bars(S5[pair],START,END);outcomes.update(arm_outcomes(pair,by_pair[pair],bars,lookup))
 summaries={};all_receipts=[]
 for arm in ARMS:summaries[arm],rows=summarize(arm,signals,outcomes);all_receipts+=rows
 gate={arm:{"minimum_completed_cycles":v["completed_cycles"]>=100,"positive_expectancy":bool(v["expectancy_per_completed_cycle_jpy"] and v["expectancy_per_completed_cycle_jpy"]>0),"profit_factor_gt_1":v["profit_factor"] is None or v["profit_factor"]>1,"unresolved_zero":v["unresolved_inventory"]==0,"margin_closeouts_zero":v["margin_closeouts"]==0} for arm,v in summaries.items()}
 result={"contract":"OPERATOR_ALPHA_OOS_SUMMARY_ALL_V1","canonical_summary":True,"runner":"research_local_equivalent","standard_invariants":{"hard_tp_enabled":True,"hard_sl_enabled":False,"end_of_replay_forced_close_excluded":True,"summary_file":"summary_all.json"},"signal_manifest_sha256":file_sha(ROOT/"signal_manifest_v1.json"),"selected_trigger":manifest["selected_trigger"],"arms":summaries,"adoption_gates":gate,"adoption":"HOLD"}
 (ROOT/"decision_receipts_v1.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in all_receipts));(ROOT/"summary_all.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2));return result
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--build-signals",action="store_true");p.add_argument("--replay",action="store_true");a=p.parse_args()
 if a.build_signals:build_signals()
 if a.replay:replay()
 if not (a.build_signals or a.replay):p.error("choose --build-signals or --replay")
 return 0
if __name__=="__main__":raise SystemExit(main())

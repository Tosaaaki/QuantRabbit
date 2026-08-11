#!/usr/bin/env python3
"""Research-only fixed-gross fast-rotation replay; no execution client exists here."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT=Path(__file__).resolve().parent
ARMS={"FIXED_GROSS_500":{"kind":"JPY","value":500.0},"ATR_010":{"kind":"ATR","value":0.10},"ATR_020":{"kind":"ATR","value":0.20},"ATR_025":{"kind":"ATR","value":0.25}}

def parse(v:str)->datetime:
    v=v.replace("Z","+00:00")
    if "." in v:
        h,r=v.split(".",1); f,z=r.split("+",1); v=f"{h}.{f[:6]}+{z}"
    return datetime.fromisoformat(v).astimezone(timezone.utc)
def iso(v:datetime)->str:return v.astimezone(timezone.utc).isoformat().replace("+00:00","Z")
def mid(r:dict[str,Any],k:str="c")->float:return (float(r["bid"][k])+float(r["ask"][k]))/2
def executable(r:dict[str,Any],side:str,k:str)->float:return float(r["ask"][k] if side=="SHORT" else r["bid"][k])
def jpy_q(pair:str,entry:float)->float:return 1.0 if pair.endswith("JPY") else 159.0
def pnl(side:str,entry:float,exit_:float,units:float,q:float)->float:return (exit_-entry)*units*q*(1 if side=="LONG" else -1)
def sha(v:Any)->str:return hashlib.sha256((json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
def ema(xs:list[float],n:int)->float:
    a=2/(n+1); out=xs[0]
    for x in xs[1:]:out=a*x+(1-a)*out
    return out
def atr_m1(rows:list[dict[str,Any]], entry_at:datetime)->float|None:
    before=[r for r in rows if parse(r["time"])+timedelta(minutes=1)<=entry_at]
    if len(before)<15:return None
    values=[]; prior=mid(before[-15])
    for r in before[-14:]:
        values.append(max(float(r["bid"]["h"]),float(r["ask"]["h"]))-min(float(r["bid"]["l"]),float(r["ask"]["l"]),prior));prior=mid(r)
    return mean(values)
def h4_direction(rows:list[dict[str,Any]], entry_at:datetime)->str:
    xs=[mid(r) for r in rows if parse(r["time"])+timedelta(hours=4)<=entry_at]
    if len(xs)<22:return "UNKNOWN"
    f,s=ema(xs,8),ema(xs,21); return "LONG" if f>s and xs[-1]>s else "SHORT" if f<s and xs[-1]<s else "NEUTRAL"
def m1_trigger(rows:list[dict[str,Any]],entry_at:datetime,side:str)->bool:
    xs=[mid(r) for r in rows if parse(r["time"])+timedelta(minutes=1)<=entry_at]
    if len(xs)<6:return False
    # completed bars only: a two-bar pullback followed by a directional reclaim.
    return xs[-1]>xs[-2] and xs[-2]<xs[-3] if side=="LONG" else xs[-1]<xs[-2] and xs[-2]>xs[-3]
def target_price(side:str,entry:float,units:float,q:float,atr:float,arm:dict[str,float])->float:
    delta=arm["value"]/(units*q) if arm["kind"]=="JPY" else atr*arm["value"]
    return entry+delta if side=="LONG" else entry-delta
def replay_one(trade:dict[str,Any], candles:dict[tuple[str,str],list[dict[str,Any]]], arm_name:str, arm:dict[str,float])->dict[str,Any]:
    entry_at=parse(trade["entry_time"]); side=trade["side"]; pair=trade["pair"]; units=trade["units"]; entry=trade["entry_price"];q=trade["q"]
    s5=candles[(trade["entry_fill_id"],"S5")];m1=candles[(trade["entry_fill_id"],"M1")];h4=candles[(trade["entry_fill_id"],"H4")]
    atr=atr_m1(m1,entry_at); htf=h4_direction(h4,entry_at); trigger=m1_trigger(m1,entry_at,side)
    base={"decision_id":f"FGR500-{trade['entry_fill_id']}","cohort_id":trade["cohort_id"],"arm":arm_name,"pair":pair,"side":side,"units":units,"entry_utc":iso(entry_at),"entry_price":entry,"htf_direction":htf,"htf_aligned":htf==side,"m1_trigger":trigger,"s5_input_available":bool(s5),"atr_m1":atr,"concurrency":1,"entry_probability":1.0,"entry_units_intent":units}
    if atr is None:return {**base,"status":"NOT_EVALUABLE_MISSING_ATR","terminal_contribution_jpy":None}
    target=target_price(side,entry,units,q,atr,arm); first_full=entry_at.replace(microsecond=0)+timedelta(seconds=5)
    horizon=entry_at+timedelta(minutes=60); last=None
    for r in s5:
        ts=parse(r["time"])
        if ts<first_full or ts>horizon:continue
        last=r
        hit=float(r["bid"]["h"] if side=="LONG" else r["ask"]["l"])>=target if side=="LONG" else float(r["ask"]["l"])<=target
        if hit:
            spread=float(r["ask"]["c"])-float(r["bid"]["c"]); slip=spread*0.10*units*q
            gross=pnl(side,entry,target,units,q); net=gross-slip
            return {**base,"status":"TAKE_PROFIT","target_price":target,"gross_jpy":gross,"after_cost_jpy":net,"terminal_contribution_jpy":net,"exit_utc":iso(ts),"holding_seconds":(ts-entry_at).total_seconds(),"exit_slippage_jpy":slip,"flat_confirmation":"SIMULATED_BROKER_FLAT_CONFIRMED_NEXT_COMPLETED_BAR","reentry_eligible":True,"state_trace":["SCAN","WAIT_DIRECTION","WAIT_TRIGGER","SHADOW_OPEN","TAKE_PROFIT","FLAT_PENDING","FLAT_CONFIRMED","REENTRY_ELIGIBLE"]}
    if last is None:return {**base,"status":"NOT_EVALUABLE_NO_S5","terminal_contribution_jpy":None}
    mtm=pnl(side,entry,executable(last,side,"c"),units,q)
    return {**base,"status":"UNRESOLVED_MTM","target_price":target,"gross_jpy":None,"after_cost_jpy":mtm,"terminal_contribution_jpy":mtm,"exit_utc":None,"holding_seconds":(parse(last["time"])-entry_at).total_seconds(),"max_monitor_minutes":60,"final_executable_mtm_jpy":mtm,"flat_confirmation":None,"reentry_eligible":False,"state_trace":["SCAN","WAIT_DIRECTION","WAIT_TRIGGER","SHADOW_OPEN"]}
def load()->tuple[list[dict[str,Any]],dict[tuple[str,str],list[dict[str,Any]]]]:
    tx=json.loads((ROOT/"source_transactions_v1.json").read_text()); c=json.loads((ROOT/"source_candles_v1.json").read_text())
    by={x["id"]:x for x in tx["transactions"]};rows=[]
    for x in tx["trades"]:
        e,cl=by[x["entry_fill_id"]],by[x["close_fill_id"]]; units=abs(int(e["units"])); side="LONG" if int(e["units"])>0 else "SHORT"; pair=e["instrument"]
        rows.append({"entry_fill_id":x["entry_fill_id"],"cohort_id":f"{x['entry_fill_id']}->{x['close_fill_id']}","label":x["label"],"entry_time":e["time"],"entry_price":float(e["price"]),"pair":pair,"units":units,"side":side,"q":jpy_q(pair,float(e["price"])),"actual_close_jpy":float(cl.get("pl") or 0),"actual_close_reason":cl.get("reason")})
    packs={(p["entry_fill_id"],p["granularity"]):p["rows"] for p in c["packets"]};return rows,packs
def metrics(rows:list[dict[str,Any]])->dict[str,Any]:
    evals=[r for r in rows if r["terminal_contribution_jpy"] is not None]; values=[r["terminal_contribution_jpy"] for r in evals]; wins=[x for x in values if x>0]; losses=[-x for x in values if x<0]
    return {"decisions":len(rows),"evaluable":len(evals),"take_profit":sum(r["status"]=="TAKE_PROFIT" for r in rows),"unresolved":sum(r["status"]=="UNRESOLVED_MTM" for r in rows),"after_cost_terminal_jpy":sum(values),"expectancy_jpy":mean(values) if values else None,"profit_factor":sum(wins)/sum(losses) if losses else None,"max_equity_drawdown_jpy":max(0.0,-min(__import__('itertools').accumulate(values,initial=0.0))),"mean_holding_seconds":mean([r["holding_seconds"] for r in rows if r.get("holding_seconds") is not None]) if any(r.get("holding_seconds") is not None for r in rows) else None,"htf_aligned":sum(bool(r["htf_aligned"]) for r in rows),"m1_triggered":sum(bool(r["m1_trigger"]) for r in rows)}
def run()->dict[str,Any]:
    trades,candles=load(); all_rows=[]; report={}
    for name,arm in ARMS.items():
        rows=[replay_one(t,candles,name,arm) for t in trades]; all_rows+=rows; report[name]=metrics(rows)
    clean=[r for r in all_rows if r["arm"]=="FIXED_GROSS_500" and "overlapping" not in next(t["label"] for t in trades if r["cohort_id"]==t["cohort_id"])]
    supply={"contract":"OPERATOR_ALPHA_SIGNAL_SUPPLY_V1","required_completed_trades":100,"definition":"one closed shadow trade needs one completed HTF-aligned S5/M1 signal, one flat confirmation, and concurrency slot zero","formula":{"required_confirmed_signals":"ceil(100 / observed_flat_confirmed_completion_rate)","required_calendar_days":"ceil(required_confirmed_signals / observed_eligible_signals_per_day)"},"current_manual_evidence":{"clean_fixed_500_take_profit":sum(r["status"]=="TAKE_PROFIT" for r in clean),"clean_fixed_500_total":len(clean),"eligible_signals_per_day":None,"status":"NOT_EVALUABLE_NO_FROZEN_OUT_OF_SAMPLE_SIGNAL_LOG"},"guarantee":False}
    out={"contract":"OPERATOR_ALPHA_FIXED_GROSS_500_SHADOW_REPLAY_V1","adoption":"HOLD","permissions":{"live":False,"paper":False,"broker_mutation":False,"orders":False,"deploy":False},"same_manual_cohort":[t["cohort_id"] for t in trades],"arms":report,"new_manual_trade_classification":{"cohort_id":"473212->473218","actual_close_jpy":560.0,"classification":"OVERLAPPING_HEDGE_CONTEXT_NOT_CONCURRENCY_ONE_SUCCESS","open_original_trade_at_fresh_snapshot":"473207"},"signal_supply":supply,"rows":all_rows}
    (ROOT/"decision_results_v1.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in all_rows),encoding="utf-8")
    (ROOT/"comparison_v1.json").write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (ROOT/"signal_supply_contract_v1.json").write_text(json.dumps(supply,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return out
if __name__=="__main__": print(json.dumps(run()["arms"],ensure_ascii=False,indent=2))

#!/usr/bin/env python3
"""Deterministic, offline-only replay for M5_EMA_DIRECTION_POST_ENTRY_V1."""
from __future__ import annotations

import argparse, datetime as dt, gzip, hashlib, json, math, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "preregistration.json"
RESULT = ROOT / "result.json"
PACKET = ROOT / "evidence_packet.json"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def parse_time(s):
    # Python 3.10 accepts microseconds, while the immutable OANDA rows carry ns.
    body=s[:-1] if s.endswith("Z") else s
    if "." in body:
        head,frac=body.split(".",1); body=head+"."+frac[:6]
    return dt.datetime.fromisoformat(body+"+00:00")
def pip_size(pair): return 0.01 if pair.endswith("_JPY") else 0.0001
def quantile(vals, q):
    s = sorted(vals)
    if not s: return 0.0
    x = (len(s)-1)*q; lo = int(math.floor(x)); hi = int(math.ceil(x))
    return s[lo] if lo == hi else s[lo]*(hi-x)+s[hi]*(x-lo)


def load_inputs(prereg):
    out = {}
    for pair, spec in prereg["inputs"]["files"].items():
        path = Path(spec["path"])
        if sha_file(path) != spec["sha256"]: raise ValueError(f"hash mismatch: {pair}")
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line); r["_time"] = parse_time(r["time"])
                if not r.get("complete") or r.get("granularity") != "M5" or r.get("price") != "BA":
                    raise ValueError(f"non-completed M5 BA row: {pair}")
                rows.append(r)
        if len(rows) != spec["rows"]: raise ValueError(f"row mismatch: {pair}")
        if any(rows[i]["_time"] >= rows[i+1]["_time"] for i in range(len(rows)-1)):
            raise ValueError(f"non-monotonic chronology: {pair}")
        out[pair] = rows
    return out


def bar_hash(r):
    return sha_bytes(canonical({k:v for k,v in r.items() if k != "_time"}))


def make_signals(pair, rows, prereg_sha):
    a3 = 2/4; a12 = 2/13; e3 = e12 = None; signals=[]
    for i,r in enumerate(rows):
        mid=(r["bid"]["c"]+r["ask"]["c"])/2
        e3=mid if e3 is None else a3*mid+(1-a3)*e3
        e12=mid if e12 is None else a12*mid+(1-a12)*e12
        if i < 11 or i+1 >= len(rows) or e3 == e12: continue
        side="LONG" if e3>e12 else "SHORT"
        material=f"{prereg_sha}|{pair}|{r['time']}|{bar_hash(r)}|{side}".encode()
        signals.append({"signal_id":sha_bytes(material),"pair":pair,"decision_index":i,
                        "decision_time":r["time"],"fill_index":i+1,"side":side})
    return signals


def path_pips(pair, side, entry_mid, future_mid):
    return ((future_mid-entry_mid) if side=="LONG" else (entry_mid-future_mid))/pip_size(pair)


FAMILY_SIZE = 12
FAMILY_ALPHA = 0.05


def tuning_tp(pair, rows, signals, tuning_end):
    mfes=[]; spreads=[]
    for s in signals:
        # A full 24-bar label must remain wholly inside tuning. Truncating the
        # label at the boundary would censor it; crossing the boundary leaks.
        if s["fill_index"] + 24 > tuning_end: continue
        e=rows[s["fill_index"]]
        entry_exec=e["ask"]["o"] if s["side"]=="LONG" else e["bid"]["o"]
        best=0.0
        for j in range(s["fill_index"], s["fill_index"]+24):
            r=rows[j]
            exit_exec=r["bid"]["c"] if s["side"]=="LONG" else r["ask"]["c"]
            best=max(best,path_pips(pair,s["side"],entry_exec,exit_exec))
        mfes.append(best)
        x=rows[s["fill_index"]+23]
        entry_half=(e["ask"]["o"]-e["bid"]["o"])/(2*pip_size(pair))
        exit_half=(x["ask"]["c"]-x["bid"]["c"])/(2*pip_size(pair))
        spreads.append(entry_half+exit_half)
    if not mfes or not spreads: raise ValueError(f"insufficient bounded tuning labels: {pair}")
    return max(quantile(mfes,.4),1.25*statistics.median(spreads))


def exit_trade(pair, rows, signal_by_index, s, arm, max_age, tp):
    fi=s["fill_index"]; entry=rows[fi]; em=(entry["bid"]["o"]+entry["ask"]["o"])/2
    side=s["side"]; entry_exec=entry["ask"]["o"] if side=="LONG" else entry["bid"]["o"]
    peak_close=0.0; mfe=-1e99; mae=1e99; reason="MAX_AGE"; exec_override=None
    last=min(fi+max_age-1,len(rows)-1); terminal=last==len(rows)-1 and fi+max_age-1>=len(rows)-1
    exit_i=last; raw_exit=None; exit_at="CLOSE"; mtm_points=[]
    for j in range(fi,last+1):
        r=rows[j]
        executable_high=r["bid"]["h"] if side=="LONG" else r["ask"]["l"]
        executable_low=r["bid"]["l"] if side=="LONG" else r["ask"]["h"]
        favorable=path_pips(pair,side,entry_exec,executable_high)
        adverse=path_pips(pair,side,entry_exec,executable_low)
        mfe=max(mfe,favorable); mae=min(mae,adverse)
        close=(r["bid"]["c"]+r["ask"]["c"])/2
        executable_close=r["bid"]["c"] if side=="LONG" else r["ask"]["c"]
        cp=path_pips(pair,side,entry_exec,executable_close)
        raw_mark=path_pips(pair,side,em,close)
        observed_mark=path_pips(pair,side,entry_exec,executable_close)
        mtm_points.append({"time":r["time"],"raw":raw_mark,"base":observed_mark-.6,"adverse":observed_mark-1.8})
        peak_close=max(peak_close,cp)
        if j < last:
            next_reason=None
            if arm in ("C","D") and cp>=tp: next_reason="TP_CLOSE_TRIGGER_NEXT_OPEN"
            elif arm=="D" and peak_close>0 and cp<=.5*peak_close: next_reason="GIVEBACK_NEXT_OPEN"
            elif arm=="B" and j>fi and j in signal_by_index and signal_by_index[j]["side"]!=side: next_reason="OPPOSITE_NEXT_OPEN"
            if next_reason:
                exit_i=j+1; x=rows[exit_i]
                raw_exit=(x["bid"]["o"]+x["ask"]["o"])/2
                exec_override=x["bid"]["o"] if side=="LONG" else x["ask"]["o"]
                reason=next_reason; exit_at="OPEN"; break
    if raw_exit is None:
        r=rows[exit_i]; raw_exit=(r["bid"]["c"]+r["ask"]["c"])/2
        if terminal: reason="TERMINAL_LIQUIDATION"
    return exit_i,raw_exit,mfe,mae,reason,exec_override,exit_at,mtm_points


def replay_config(pair, rows, signals, split_start, arm, max_age, tp):
    eligible=[]
    for s in signals:
        if s["fill_index"]<split_start: continue
        fi=s["fill_index"]; horizon=min(fi+max_age-1,len(rows)-1)
        e=rows[fi]; em=(e["bid"]["o"]+e["ask"]["o"])/2
        hm=(rows[horizon]["bid"]["c"]+rows[horizon]["ask"]["c"])/2
        eligible.append(dict(s,direction_correct=path_pips(pair,s["side"],em,hm)>0))
    sigidx={s["decision_index"]:s for s in signals}; trades=[]; busy_until=-1; skips=0
    for s in eligible:
        if s["fill_index"]<=busy_until: skips+=1; continue
        fi=s["fill_index"]; e=rows[fi]; em=(e["bid"]["o"]+e["ask"]["o"])/2
        xi,xm,mfe,mae,reason,exec_override,exit_at,mtm_points=exit_trade(pair,rows,sigidx,s,arm,max_age,tp); x=rows[xi]
        raw=path_pips(pair,s["side"],em,xm)
        entry_exec=e["ask"]["o"] if s["side"]=="LONG" else e["bid"]["o"]
        exit_exec=exec_override if exec_override is not None else (x["bid"]["c"] if s["side"]=="LONG" else x["ask"]["c"])
        observed=path_pips(pair,s["side"],entry_exec,exit_exec)
        base=observed-.6; adverse=observed-1.8
        trades.append({"signal_id":s["signal_id"],"pair":pair,"side":s["side"],"entry_index":fi,"exit_index":xi,
          "entry_time":e["time"],"exit_time":x["time"],"exit_reason":reason,"age_bars":xi-fi+1,
          "entry_raw":em,"entry_executable":entry_exec,"exit_raw":xm,"exit_executable":exit_exec,
          "raw_pips":raw,"base_pips":base,"adverse_pips":adverse,"mfe_pips":mfe,"mae_pips":mae,
          "direction_correct":s["direction_correct"],"roundtrip_spread_pips":raw-observed,
          "terminal_liquidation":reason=="TERMINAL_LIQUIDATION","exit_at":exit_at,"mtm_points":mtm_points})
        busy_until=xi
    return eligible,trades,skips


def jpy_pnl_at(pair,pips,when,usd_jpy_rows):
    units=1000; ps=pip_size(pair)
    quote=units*pips*ps
    if pair.endswith("_JPY"): return quote
    t=parse_time(when); lo=0; hi=len(usd_jpy_rows)
    while lo<hi:
        m=(lo+hi)//2
        if usd_jpy_rows[m]["_time"]<=t: lo=m+1
        else: hi=m
    rate=(usd_jpy_rows[max(0,lo-1)]["bid"]["c"]+usd_jpy_rows[max(0,lo-1)]["ask"]["c"])/2
    return quote*rate


def jpy_pnl(trade,pips,usd_jpy_rows):
    return jpy_pnl_at(trade["pair"],pips,trade["exit_time"],usd_jpy_rows)


def metrics(signals,trades,skips,scenario,usd_jpy_rows):
    vals=[t[f"{scenario}_pips"] for t in trades]; n=len(vals)
    pnl=[jpy_pnl(t,v,usd_jpy_rows) for t,v in zip(trades,vals)]
    equity=200000.; clusters={}; realized_by_month={}; exit_events={}; mark_events={}
    for t,p,v in zip(trades,pnl,vals):
        equity+=p
        month=t["exit_time"][:7]; realized_by_month[month]=realized_by_month.get(month,0.)+p
        key=t["exit_time"][:10]; clusters.setdefault(key,[]).append(v)
        exit_events.setdefault(t["exit_time"],[]).append((t,p))
        for point in t["mtm_points"]:
            mark=jpy_pnl_at(t["pair"],point[scenario],point["time"],usd_jpy_rows)
            mark_events.setdefault(point["time"],[]).append((t["signal_id"],mark))
    cmeans=[statistics.mean(x) for x in clusters.values()]
    se=statistics.stdev(cmeans)/math.sqrt(len(cmeans)) if len(cmeans)>1 else float("inf")
    family_z=statistics.NormalDist().inv_cdf(1-FAMILY_ALPHA/(2*FAMILY_SIZE))
    cluster_mean=statistics.mean(cmeans) if cmeans else None
    lcb=cluster_mean-family_z*se if cluster_mean is not None and math.isfinite(se) else None

    realized=0.; active={}; peak=200000.; maxdd=0.; month_end_equity={}; ruin_time=None
    timeline=sorted(set(mark_events)|set(exit_events))
    for when in timeline:
        # Close positions scheduled for this bar open before consuming that
        # bar's completed-close marks.
        for t,p in exit_events.get(when,[]):
            if t["exit_at"]=="OPEN":
                realized+=p; active.pop(t["signal_id"],None)
        for sid,mark in mark_events.get(when,[]): active[sid]=mark
        marked=200000.+realized+sum(active.values())
        peak=max(peak,marked); maxdd=min(maxdd,marked/peak-1)
        if ruin_time is None and marked<=0: ruin_time=when
        for t,p in exit_events.get(when,[]):
            if t["exit_at"]=="CLOSE":
                realized+=p; active.pop(t["signal_id"],None)
        settled=200000.+realized+sum(active.values())
        peak=max(peak,settled); maxdd=min(maxdd,settled/peak-1)
        if ruin_time is None and settled<=0: ruin_time=when
        month_end_equity[when[:7]]=settled
    monthly={}; monthly_change_initial={}; prior=200000.
    for month,end_equity in sorted(month_end_equity.items()):
        monthly[month]=(end_equity/prior) if prior>0 else None
        monthly_change_initial[month]=1+realized_by_month.get(month,0.)/200000.
        prior=end_equity
    valid_monthly=[v for v in monthly.values() if v is not None]
    return {"raw_signals":len(signals),"trades":n,"collision_skips":skips,
      "direction_accuracy":sum(s["direction_correct"] for s in signals)/len(signals) if signals else 0,
      "executed_direction_accuracy":sum(t["direction_correct"] for t in trades)/n if n else 0,
      "expectancy_pips":statistics.mean(vals) if n else 0,"cluster_mean_expectancy_pips":cluster_mean,
      "family_adjusted_lcb_pips":lcb,"family_critical_z":family_z,"lcb_proxy_pips":lcb,
      "gross_expectancy_pips":statistics.mean([t["raw_pips"] for t in trades]) if n else 0,
      "break_even_roundtrip_cost_pips":statistics.mean([t["raw_pips"] for t in trades]) if n else 0,
      "mfe_mean_pips":statistics.mean([t["mfe_pips"] for t in trades]) if n else 0,"mae_mean_pips":statistics.mean([t["mae_pips"] for t in trades]) if n else 0,
      "cost_drag_pips":statistics.mean([t["raw_pips"]-t[f"{scenario}_pips"] for t in trades]) if n else 0,
      "turnover_units":2*1000*n,"inventory_age_mean_bars":statistics.mean([t["age_bars"] for t in trades]) if n else 0,
      "inventory_age_max_bars":max([t["age_bars"] for t in trades],default=0),
      "terminal_liquidation_pips":sum(t[f"{scenario}_pips"] for t in trades if t["terminal_liquidation"]),"terminal_open_inventory":0,
      "equity_multiple":equity/200000.,"return_on_initial_equity":equity/200000.-1,"max_drawdown":maxdd,
      "drawdown_basis":"completed_bar_portfolio_mtm_including_open_inventory",
      "equity_ruin":ruin_time is not None,"equity_ruin_time":ruin_time,
      "monthly_multiples":monthly,"monthly_realized_change_on_initial_equity":monthly_change_initial,
      "monthly_multiple_std":statistics.pstdev(valid_monthly) if len(valid_monthly)>1 else None,
      "n_eff_utc_day_clusters":len(cmeans)}


def aggregate(per_pair, scenario, usdrows):
    sig=[]; trades=[]; skips=0
    for v in per_pair.values(): sig+=v[0]; trades+=v[1]; skips+=v[2]
    trades.sort(key=lambda t:(t["exit_time"],t["pair"],t["signal_id"]))
    return metrics(sig,trades,skips,scenario,usdrows)


def main(write=True):
    prereg=json.loads(PREREG.read_text()); pregsha=sha_file(PREREG); data=load_inputs(prereg)
    signals={p:make_signals(p,r,pregsha) for p,r in data.items()}; splits={p:int(len(r)*.7) for p,r in data.items()}
    tps={p:tuning_tp(p,data[p],signals[p],splits[p]) for p in data}; configs={};
    for age in (6,12,24):
      for arm in "ABCD":
        cid=f"{arm}_H{age}"; tuning={}; walk={}
        tpp={p:replay_config(p,data[p][:splits[p]], [s for s in signals[p] if s['fill_index']<splits[p]],0,arm,age,tps[p]) for p in data}
        wpp={p:replay_config(p,data[p],signals[p],splits[p],arm,age,tps[p]) for p in data}
        for sc in ("raw","base","adverse"):
            tuning[sc]=aggregate(tpp,sc,data["USD_JPY"]); walk[sc]=aggregate(wpp,sc,data["USD_JPY"])
        configs[cid]={"arm":arm,"max_age_bars":age,"tuning":tuning,"walk_forward":walk}
    def score(item):
        cid,v=item; m=v["tuning"]["base"]; l=m["lcb_proxy_pips"] if m["lcb_proxy_pips"] is not None else -1e99
        month_std=m["monthly_multiple_std"] if m["monthly_multiple_std"] is not None else float("inf")
        return (-l,abs(m["max_drawdown"]),month_std,cid)
    selected=sorted(configs.items(),key=score)[0][0]; wf=configs[selected]["walk_forward"]
    months=wf["adverse"]["monthly_multiples"]
    walk_start=max(data[p][splits[p]]["_time"] for p in data); walk_end=min(data[p][-1]["_time"] for p in data)
    full_months=0
    cursor=dt.datetime(walk_start.year,walk_start.month,1,tzinfo=dt.timezone.utc)
    if cursor<walk_start:
        cursor=(cursor.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
    while cursor<=walk_end:
        nxt=(cursor.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
        if nxt-dt.timedelta(minutes=5)<=walk_end: full_months+=1
        cursor=nxt
    gates={"adverse_family_adjusted_lcb_positive":wf["adverse"]["lcb_proxy_pips"] is not None and wf["adverse"]["lcb_proxy_pips"]>0,
      "base_expectancy_positive":wf["base"]["expectancy_pips"]>0,
      "worst_month_gte_1":bool(months) and all(v is not None and v>=1 for v in months.values()),
      "full_calendar_months_gte_3":full_months>=3,"max_drawdown_gte_minus_10pct":wf["adverse"]["max_drawdown"]>=-.1,
      "no_equity_ruin":not wf["adverse"]["equity_ruin"]}
    admitted=all(gates.values())
    result={"schema_version":2,"candidate_id":prereg["candidate_id"],"prereg_sha256":pregsha,"script_sha256":sha_file(Path(__file__)),
      "input_verification":{p:{"sha256":sha_file(Path(prereg['inputs']['files'][p]['path'])),"rows":len(data[p]),"first":data[p][0]['time'],"last":data[p][-1]['time']} for p in data},
      "split_indices":splits,"walk_forward_common_coverage_utc":[walk_start.isoformat(),walk_end.isoformat()],"walk_forward_full_calendar_months":full_months,
      "raw_signal_counts":{p:len(signals[p]) for p in signals},"tp_pips_frozen_from_tuning":tps,"configs":configs,
      "selected_config":selected,"selection_basis":"tuning_executable_base_only_lexicographic_not_adverse_not_2x","selected_walk_forward":wf,"admission_gates":gates,"admission":admitted,
      "accounting_model":prereg["portfolio"],"profit_unproven":not admitted,"holdout_unopened":True,"external_orders":0,"live_authority":False}
    if write:
        RESULT.write_bytes(json.dumps(result,sort_keys=True,indent=2).encode()+b"\n")
        packet={"schema_version":2,"candidate_id":prereg["candidate_id"],"status":"ADMITTED" if admitted else "UNADMITTED_RESEARCH_RESULT",
          "exact_config":{"config_id":selected,"arm":configs[selected]["arm"],"max_age_bars":configs[selected]["max_age_bars"],
            "tp_pips_by_pair":tps,"costs":prereg["costs"],"portfolio":prereg["portfolio"]},
          "config_results":configs[selected],"formula":prereg["signal"],"sources":prereg["inputs"]["files"],"prereg_sha256":pregsha,
          "script_sha256":result["script_sha256"],"test_sha256":sha_file(ROOT/"test_replay_m5_ema.py"),
          "readme_sha256":sha_file(ROOT/"README.md"),"invalidated_draft_sha256":sha_file(ROOT/"INVALIDATED_DRAFT_PRESEAL.json"),
          "result_sha256":sha_file(RESULT),"selected_config":selected,"admission":admitted,
          "profit_unproven":not admitted,"holdout_unopened":True,"external_orders":0}
        PACKET.write_bytes(json.dumps(packet,sort_keys=True,indent=2).encode()+b"\n")
    return result


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--no-write",action="store_true"); a=ap.parse_args(); r=main(not a.no_write)
    print(json.dumps({"selected_config":r["selected_config"],"admission":r["admission"],"walk_forward":r["selected_walk_forward"]},sort_keys=True))

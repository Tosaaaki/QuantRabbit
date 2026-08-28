"""Transactional, credential-free, zero-authority FX shadow core."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "2026-08-28-p1"
TIMEFRAMES = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400}
UNIVERSE = ("AUD_USD","EUR_USD","GBP_USD","NZD_USD","USD_CAD","USD_CHF","USD_JPY")
LEDGERS = ("raw_events","feed_quality","completed_bars","policy_receipts","proposals",
           "expected_orders","virtual_fills","pnl","batch_manifests","control")
ZERO_CAPS = ("network_attempts","credential_reads","external_order_attempts","external_orders")


class IntegrityError(RuntimeError): pass
class InjectedCrash(RuntimeError): pass


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def canonical_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path):
    return hashlib.sha256(secure_read(Path(path))).hexdigest()


def parse_utc(value):
    stamp=datetime.fromisoformat(value.replace("Z","+00:00"))
    if stamp.tzinfo is None: raise ValueError("aware UTC required")
    return stamp.astimezone(timezone.utc)


def utc_text(value):
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")


def real_dir(path):
    path.mkdir(parents=True,exist_ok=True)
    if path.is_symlink() or not path.is_dir(): raise IntegrityError(f"unsafe directory: {path}")


def valid_target(path, missing=True):
    real_dir(path.parent)
    try: st=os.lstat(path)
    except FileNotFoundError:
        if missing: return
        raise
    if os.path.islink(path) or not os.path.isfile(path) or st.st_nlink != 1:
        raise IntegrityError(f"unsafe file: {path}")


def secure_read(path):
    valid_target(path,False); before=os.lstat(path)
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    try:
        fcntl.flock(fd,fcntl.LOCK_SH)
        after=os.fstat(fd)
        if (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino) or after.st_nlink!=1:
            raise IntegrityError("swap/hardlink")
        chunks=[]
        while True:
            chunk=os.read(fd,1048576)
            if not chunk: break
            chunks.append(chunk)
        final=os.fstat(fd)
        if (after.st_dev,after.st_ino,after.st_size)!=(final.st_dev,final.st_ino,final.st_size):
            raise IntegrityError("changed during read")
        return b"".join(chunks)
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


def secure_append(path,data):
    valid_target(path)
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT|getattr(os,"O_NOFOLLOW",0),0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        if os.fstat(fd).st_nlink != 1: raise IntegrityError("hardlink append")
        view=memoryview(data)
        while view: view=view[os.write(fd,view):]
        os.fsync(fd)
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


def atomic_json(path,payload):
    valid_target(path)
    data=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n"
    tmp=path.parent/f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
    try:
        view=memoryview(data)
        while view: view=view[os.write(fd,view):]
        os.fsync(fd)
    finally: os.close(fd)
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class JsonlFixtureAdapter:
    feed_identity="LOCAL_FIXTURE_BBO_V1"; continuity="SEQUENCED"; lossless=True
    def __init__(self,path,pace_seconds=0): self.path=Path(path); self.pace_seconds=pace_seconds
    def source_bytes(self): return secure_read(self.path)
    def events(self):
        for line in self.source_bytes().decode("utf-8","strict").splitlines():
            if line.strip():
                if self.pace_seconds: time.sleep(self.pace_seconds)
                yield json.loads(line)


class NonSequencedFixtureAdapter(JsonlFixtureAdapter):
    feed_identity="LOCAL_FIXTURE_BBO_NONSEQUENCED_V1"; continuity="HEARTBEAT_ONLY"; lossless=False


class HashLedger:
    def __init__(self,path):
        self.path=Path(path); self.rows=[]; self.by_id={}; self.last_hash="0"*64; self.byte_size=0
        if self.path.exists() or self.path.is_symlink(): self.verify()
    def verify(self):
        data=secure_read(self.path); previous="0"*64; seen=set(); rows=[]; by_id={}
        for index,line in enumerate(data.decode("utf-8","strict").splitlines(),1):
            row=json.loads(line); unsigned={k:v for k,v in row.items() if k!="record_hash"}
            if row.get("sequence_no")!=index or row.get("previous_hash")!=previous or canonical_hash(unsigned)!=row.get("record_hash"):
                raise IntegrityError("ledger chain mismatch")
            if row.get("record_id") in seen: raise IntegrityError("duplicate record id")
            seen.add(row["record_id"]); previous=row["record_hash"]; rows.append(row); by_id[row["record_id"]]=row
        self.rows=rows; self.by_id=by_id; self.last_hash=previous; self.byte_size=len(data)
    def refresh(self):
        if not self.path.exists() and not self.path.is_symlink(): return 0
        valid_target(self.path,False); before=os.lstat(self.path)
        fd=os.open(self.path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
        try:
            fcntl.flock(fd,fcntl.LOCK_SH); current=os.fstat(fd)
            if (before.st_dev,before.st_ino)!=(current.st_dev,current.st_ino) or current.st_nlink!=1:
                raise IntegrityError("swap/hardlink")
            if current.st_size<self.byte_size: raise IntegrityError("ledger truncated")
            if current.st_size==self.byte_size: return 0
            os.lseek(fd,self.byte_size,os.SEEK_SET); chunks=[]
            while True:
                chunk=os.read(fd,1048576)
                if not chunk: break
                chunks.append(chunk)
            final=os.fstat(fd)
            if (current.st_dev,current.st_ino,current.st_size)!=(final.st_dev,final.st_ino,final.st_size):
                raise IntegrityError("changed during read")
            data=b"".join(chunks)
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)
        if not data.endswith(b"\n"): raise IntegrityError("partial ledger tail")
        previous=self.last_hash; new=[]
        for line in data.decode("utf-8","strict").splitlines():
            row=json.loads(line); unsigned={k:v for k,v in row.items() if k!="record_hash"}
            expected=len(self.rows)+len(new)+1
            if row.get("sequence_no")!=expected or row.get("previous_hash")!=previous or canonical_hash(unsigned)!=row.get("record_hash"):
                raise IntegrityError("ledger tail mismatch")
            if row.get("record_id") in self.by_id or any(x["record_id"]==row.get("record_id") for x in new):
                raise IntegrityError("duplicate record id")
            previous=row["record_hash"]; new.append(row)
        self.rows.extend(new)
        for row in new: self.by_id[row["record_id"]]=row
        self.last_hash=previous; self.byte_size=final.st_size
        return len(new)
    def plan(self,payload,record_id=None,planned=None):
        planned=[] if planned is None else planned
        payload=copy.deepcopy(payload)
        record_id=record_id or canonical_hash({"ledger":self.path.name,"payload":payload})
        existing=self.by_id.get(record_id)
        if existing is not None:
            if existing["payload"]!=payload: raise IntegrityError("record id conflict")
            return existing
        for row in planned:
            if row["record_id"]==record_id:
                if row["payload"]!=payload: raise IntegrityError("record id conflict")
                return row
        row={"sequence_no":len(self.rows)+len(planned)+1,
             "previous_hash":planned[-1]["record_hash"] if planned else self.last_hash,
             "record_id":record_id,"payload":payload}
        row["record_hash"]=canonical_hash(row); planned.append(row); return row
    def append_rows(self,rows):
        pending=[]
        for row in rows:
            if row["record_id"] in self.by_id:
                if self.by_id[row["record_id"]]!=row: raise IntegrityError("idempotency conflict")
                continue
            expected=len(self.rows)+len(pending)+1
            previous=pending[-1]["record_hash"] if pending else self.last_hash
            unsigned={k:v for k,v in row.items() if k!="record_hash"}
            if row["sequence_no"]!=expected or row["previous_hash"]!=previous or canonical_hash(unsigned)!=row["record_hash"]:
                raise IntegrityError("transaction boundary mismatch")
            pending.append(row)
        if pending:
            data=b"".join(canonical_bytes(r)+b"\n" for r in pending)
            secure_append(self.path,data)
            self.rows.extend(pending)
            for row in pending: self.by_id[row["record_id"]]=row
            self.last_hash=pending[-1]["record_hash"]; self.byte_size+=len(data)


class RuntimeLock:
    def __init__(self,root,strategy_hash): self.root=root; self.strategy_hash=strategy_hash; self.handle=None
    def __enter__(self):
        real_dir(self.root); path=self.root/"runtime.lock"; valid_target(path)
        fd=os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,"O_NOFOLLOW",0),0o600)
        if os.fstat(fd).st_nlink!=1: raise IntegrityError("hardlinked lock")
        self.handle=os.fdopen(fd,"r+",encoding="utf-8")
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError("duplicate writer") from exc
        self.handle.seek(0); self.handle.truncate(); self.handle.write(str(os.getpid())+"\n")
        self.handle.flush(); os.fsync(fd)
        atomic_json(self.root/"pid.json",{"pid":os.getpid(),"strategy_hash":self.strategy_hash,"started_at_utc":utc_text(datetime.now(timezone.utc))})
        return self
    def heartbeat(self,status,at=None):
        row={"schema_version":2,"pid":os.getpid(),"strategy_hash":self.strategy_hash,
             "beat_at_utc":utc_text(at or datetime.now(timezone.utc)),"run_state":status["run_state"],
             "feed_blocked":status["feed_blocked"],"counters":status["counters"]}
        row["heartbeat_hash"]=canonical_hash(row); atomic_json(self.root/"heartbeat.json",row)
    def __exit__(self,*_):
        if self.handle: fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN); self.handle.close()


class ShadowRuntime:
    def __init__(self,package_root,runtime_root):
        self.package_root=Path(package_root).resolve(); self.runtime_root=Path(runtime_root)
        real_dir(self.runtime_root); real_dir(self.runtime_root/"ledgers")
        self.packet=json.loads(secure_read(self.package_root/"baseline_packet.json"))
        self.contract=json.loads(secure_read(self.package_root/"runtime_contract.json"))
        self.packet_hash=sha256_file(self.package_root/"baseline_packet.json")
        self.contract_hash=sha256_file(self.package_root/"runtime_contract.json")
        self.strategy_hash=canonical_hash({"packet_sha256":self.packet_hash,"runtime_contract_sha256":self.contract_hash})
        self.verify_contract(); self.arms=self.contract["arms"]
        self.state_path=self.runtime_root/"state.json"; self.checkpoint_path=self.runtime_root/"checkpoint.json"
        self.pending_path=self.runtime_root/"pending_transaction.json"; self.status_path=self.runtime_root/"status.json"
        self.ledgers={n:HashLedger(self.runtime_root/"ledgers"/f"{n}.jsonl") for n in LEDGERS}
        self.recover(); self.state=self.load_state(); self.stop_requested=False
    def verify_contract(self):
        if self.packet["candidate_id"]!=self.contract["candidate_id"]: raise IntegrityError("candidate mismatch")
        if self.packet["packet_status"]!="R5_CORRECTED_STABLE_BOUND_CONDITIONAL_MASSIVE_ADAPTER_READY_UNCONNECTED": raise IntegrityError("R5 packet status")
        if self.packet["strategy_status"]!="RESEARCH_NOT_ADMITTED" or self.contract["strategy_status"]!="RESEARCH_NOT_ADMITTED": raise IntegrityError("research boundary")
        if self.packet["live_authority"]!="NONE" or self.contract["live_authority"]!="NONE": raise IntegrityError("authority boundary")
        expected_binding={
          "source_commit":"15c7d205a78a14116651046b5fa2741d37e72cf2",
          "source_files_aggregate_sha256":"5b530abf7aa7d3db0f31826efc9f4645d2a9dceb7e10feb620d40f5f3c0ed016",
          "evidence_aggregate_sha256":"027bd900ba1fbd237b15e0cbd5e393648e36bcbabd629c53d81cf2286cab94a7",
          "combined_aggregate_sha256":"a6f2c4990af89462d33dca0bee8fca0c3af700562de4f074c5f90331d3d87d13"}
        if any(self.packet["r5_binding"].get(k)!=v or self.contract["r5_binding"].get(k)!=v for k,v in expected_binding.items()):
            raise IntegrityError("R5 content binding")
        blockers=[
          "BLOCKED_NEEDS_MASSIVE_CURRENCIES_BUSINESS_CONTRACT",
          "BLOCKED_NEEDS_EXPLICIT_FEED_RECORDER_MARKET_DATA_KEY_PERMISSION"]
        feed=self.packet["market_data_contract"]
        runtime_feed=self.contract["feed_binding"]
        if feed.get("blockers")!=blockers or runtime_feed.get("blockers")!=blockers: raise IntegrityError("feed blockers")
        if feed.get("feed_binding")!="CONDITIONAL_MASSIVE" or feed.get("credential_authority")!="NONE":
            raise IntegrityError("feed authority unexpectedly enabled")
        if feed.get("primary_provider_candidate")!="MASSIVE_REALTIME_FOREX_QUOTES_WEBSOCKET" or feed.get("fallback_providers")!=[]:
            raise IntegrityError("provider candidate")
        if feed.get("feed_connected") or feed.get("foreground_observation_enabled") or feed.get("decisions_allowed"):
            raise IntegrityError("feed authority unexpectedly enabled")
        if runtime_feed.get("status")!="CONDITIONAL_MASSIVE" or runtime_feed.get("credential_authority")!="NONE":
            raise IntegrityError("runtime feed unexpectedly bound")
        if runtime_feed.get("primary_provider_candidate")!="MASSIVE_REALTIME_FOREX_QUOTES_WEBSOCKET" or runtime_feed.get("fallback_providers")!=[]:
            raise IntegrityError("runtime provider candidate")
        if runtime_feed.get("connected") or runtime_feed.get("foreground_observation_enabled"):
            raise IntegrityError("runtime feed unexpectedly enabled")
    def fresh_state(self):
        counters={k:0 for k in ("market_events_received","market_events_accepted","duplicate_events","sequence_gaps",
          "out_of_order_events","arrival_regressions","clock_drift_events","silence_expiries","completed_bars",
          "proposals","decisions","expected_orders","virtual_fills","virtual_exits","pnl_records","llm_calls",
          "invalid_intervals","durable_halts","recovered_transactions",*ZERO_CAPS)}
        return {"schema_version":2,"strategy_hash":self.strategy_hash,"contract_hash":self.contract_hash,
          "run_state":"RUNNING","feed_blocked":False,"block_reasons":[],"period_state":"OPEN",
          "seen_events":{},"last_sequence":{},"last_event_time":{},"last_arrival_time":{},
          "bar_builders":{},"daily_m5":{},"sealed_periods":{},"proposal_ids":[],
          "proposals_by_id":{},"inventories":{},"last_quotes":{},"sources":{},
          "arm_status":{a:"RUNNING" for a in self.arms},"counters":counters,
          "realized_return":{a:0.0 for a in self.arms},"unrealized_return":{a:0.0 for a in self.arms}}
    def checkpoint(self,state):
        row={"schema_version":2,"strategy_hash":self.strategy_hash,"contract_hash":self.contract_hash,
             "state_hash":canonical_hash(state),"ledger_heads":{n:x.last_hash for n,x in self.ledgers.items()},
             "ledger_rows":{n:len(x.rows) for n,x in self.ledgers.items()},
             "manifest_set_root":canonical_hash([r["record_hash"] for r in self.ledgers["batch_manifests"].rows]),
             "sealed_period_registry_root":canonical_hash(state["sealed_periods"])}
        row["checkpoint_root"]=canonical_hash(row); return row
    def load_state(self):
        if not self.state_path.exists() and not self.checkpoint_path.exists(): return self.fresh_state()
        try:
            state=json.loads(secure_read(self.state_path)); checkpoint=json.loads(secure_read(self.checkpoint_path))
            if checkpoint!=self.checkpoint(state): raise IntegrityError("checkpoint mismatch")
            if state["strategy_hash"]!=self.strategy_hash or state["contract_hash"]!=self.contract_hash: raise IntegrityError("strategy mismatch")
            return state
        except Exception as exc:
            state=self.fresh_state(); state.update(run_state="HALTED_INTEGRITY",feed_blocked=True,block_reasons=["STATE_OR_CHECKPOINT_INTEGRITY"])
            state["counters"]["durable_halts"]=1
            rows=[]; self.ledgers["control"].plan({"halt_code":"STATE_OR_CHECKPOINT_INTEGRITY","detail":type(exc).__name__},"halt::state",rows)
            self.ledgers["control"].append_rows(rows); atomic_json(self.state_path,state); atomic_json(self.checkpoint_path,self.checkpoint(state))
            return state
    def recover(self):
        if not self.pending_path.exists() and not self.pending_path.is_symlink(): return
        pending=json.loads(secure_read(self.pending_path))
        if pending["strategy_hash"]!=self.strategy_hash or pending["contract_hash"]!=self.contract_hash: raise IntegrityError("pending mismatch")
        for name in LEDGERS: self.ledgers[name].append_rows(pending["ledger_rows"][name])
        state=pending["state"]; state["counters"]["recovered_transactions"]+=1
        atomic_json(self.state_path,state); atomic_json(self.checkpoint_path,self.checkpoint(state)); os.unlink(self.pending_path)
    def plans(self): return {n:[] for n in LEDGERS}
    def add(self,plans,name,payload,record_id=None): return self.ledgers[name].plan(payload,record_id,plans[name])
    def transact(self,state,plans,fault=None):
        pending={"schema_version":2,"strategy_hash":self.strategy_hash,"contract_hash":self.contract_hash,
                 "state":state,"ledger_rows":plans}
        pending["transaction_id"]=canonical_hash(pending); atomic_json(self.pending_path,pending)
        if fault=="AFTER_PREPARE": raise InjectedCrash(fault)
        boundary=0
        for name in LEDGERS:
            self.ledgers[name].append_rows(plans[name])
            if plans[name]:
                boundary+=1
                if fault==f"AFTER_LEDGER_{boundary}": raise InjectedCrash(fault)
        atomic_json(self.state_path,state)
        if fault=="AFTER_STATE": raise InjectedCrash(fault)
        atomic_json(self.checkpoint_path,self.checkpoint(state))
        if fault=="AFTER_CHECKPOINT": raise InjectedCrash(fault)
        os.unlink(self.pending_path); self.state=state
    def durable_halt(self,code,run_state,detail=""):
        state=copy.deepcopy(self.state); plans=self.plans()
        state.update(run_state=run_state,feed_blocked=True)
        if code not in state["block_reasons"]: state["block_reasons"].append(code)
        state["counters"]["durable_halts"]+=1
        self.add(plans,"control",{"halt_code":code,"run_state":run_state,"detail":detail[:128]},f"halt::{code}::{state['counters']['durable_halts']}")
        self.transact(state,plans); self.write_status()
    def quality(self,state,event,adapter):
        required={"event_id","feed_identity","instrument","event_time_utc","arrival_time_utc","bid","ask"}
        if not required.issubset(event): return "MISSING_REQUIRED_FIELD"
        if event["feed_identity"]!=adapter.feed_identity: return "UNBOUND_FEED_IDENTITY"
        if event["instrument"] not in UNIVERSE: return "UNEXPECTED_INSTRUMENT"
        digest=canonical_hash(event)
        if event["event_id"] in state["seen_events"]: return "DUPLICATE_EVENT" if state["seen_events"][event["event_id"]]==digest else "CONFLICTING_DUPLICATE"
        try:
            bid,ask=float(event["bid"]),float(event["ask"]); source=parse_utc(event["event_time_utc"]); arrival=parse_utc(event["arrival_time_utc"])
        except Exception: return "INVALID_FIELD_TYPE"
        if not all(map(math.isfinite,(bid,ask))) or bid<=0 or ask<=bid: return "INVALID_BBO"
        instrument=event["instrument"]
        if instrument in state["last_event_time"] and source<=parse_utc(state["last_event_time"][instrument]): return "OUT_OF_ORDER_TIME"
        if instrument in state["last_arrival_time"] and arrival<=parse_utc(state["last_arrival_time"][instrument]): return "ARRIVAL_TIME_REGRESSION"
        if abs((arrival-source).total_seconds())>self.contract["continuity"]["max_arrival_clock_drift_seconds"]: return "CLOCK_DRIFT"
        if adapter.continuity=="SEQUENCED":
            if "source_sequence" not in event: return "MISSING_SEQUENCE"
            seq=int(event["source_sequence"]); previous=state["last_sequence"].get(instrument)
            if previous is not None and seq!=previous+1: return "OUT_OF_ORDER_SEQUENCE" if seq<=previous else "SEQUENCE_GAP"
        bucket=int(source.timestamp())//300*300; sealed=state["sealed_periods"].get(f"{instrument}:M5")
        if sealed and bucket<=sealed["bucket_epoch"]: return "POST_SEAL_BACKDATED_EVENT"
        return "OK"
    def seal_bar(self,state,plans,name,instrument,builder):
        bar={"timeframe":name,"instrument":instrument,**builder,"period_state":"SEALED"}
        row=self.add(plans,"completed_bars",bar,f"bar::{instrument}::{name}::{builder['bucket_epoch']}")
        bar["bar_record_hash"]=row["record_hash"]; state["counters"]["completed_bars"]+=1
        state["sealed_periods"][f"{instrument}:{name}"]={"bucket_epoch":builder["bucket_epoch"],"bar_record_hash":row["record_hash"],"summary_hash":canonical_hash(bar)}
        if name=="M5":
            state["daily_m5"].setdefault(builder["start_utc"][:10],{}).setdefault(instrument,[]).append(bar)
            self.exit_if_due(state,plans,instrument,bar)
    def aggregate(self,state,plans,event):
        stamp=parse_utc(event["event_time_utc"]); epoch=int(stamp.timestamp()); inst=event["instrument"]; bid=float(event["bid"]); ask=float(event["ask"])
        for name,seconds in TIMEFRAMES.items():
            bucket=epoch//seconds*seconds; key=f"{inst}:{name}"; b=state["bar_builders"].get(key)
            if b and b["bucket_epoch"]<bucket: self.seal_bar(state,plans,name,inst,b); b=None
            if not b:
                b={"bucket_epoch":bucket,"start_utc":utc_text(datetime.fromtimestamp(bucket,timezone.utc)),
                   "last_event_utc":event["event_time_utc"],"arrival_watermark_utc":event["arrival_time_utc"],"event_count":1,
                   "bid_o":bid,"bid_h":bid,"bid_l":bid,"bid_c":bid,"ask_o":ask,"ask_h":ask,"ask_l":ask,"ask_c":ask}
            else:
                b["event_count"]+=1; b["last_event_utc"]=event["event_time_utc"]; b["arrival_watermark_utc"]=max(b["arrival_watermark_utc"],event["arrival_time_utc"])
                b["bid_h"]=max(b["bid_h"],bid); b["bid_l"]=min(b["bid_l"],bid); b["bid_c"]=bid
                b["ask_h"]=max(b["ask_h"],ask); b["ask_l"]=min(b["ask_l"],ask); b["ask_c"]=ask
            state["bar_builders"][key]=b
    def proposal(self,state,event):
        stamp=parse_utc(event["event_time_utc"])
        if stamp.strftime("%H:%M")!="12:00": return None
        inst=event["instrument"]; day=stamp.date().isoformat()
        bars={r["start_utc"][11:16]:r for r in state["daily_m5"].get(day,{}).get(inst,[])}
        asian=[f"{m//60:02d}:{m%60:02d}" for m in range(0,360,5)]
        if any(x not in bars for x in asian) or "08:00" not in bars or "11:55" not in bars: return None
        high=max((bars[x]["bid_h"]+bars[x]["ask_h"])/2 for x in asian); low=min((bars[x]["bid_l"]+bars[x]["ask_l"])/2 for x in asian)
        opened=(bars["08:00"]["bid_o"]+bars["08:00"]["ask_o"])/2; closed=(bars["11:55"]["bid_c"]+bars["11:55"]["ask_c"])/2
        displacement=math.log(closed/opened); width=math.log(high/low)
        if displacement==0 or abs(displacement)<=width: return None
        direction=-1 if displacement>0 else 1; pid=f"LOEF::{day}::{inst}::{'LONG' if direction>0 else 'SHORT'}"
        if pid in state["proposal_ids"]: return None
        source=asian+["08:00","11:55"]
        return {"proposal_id":pid,"strategy_hash":self.strategy_hash,"instrument":inst,"direction":direction,
          "decision_bar_hash":bars["11:55"]["bar_record_hash"],"completed_bar_set_hash":canonical_hash([bars[x]["bar_record_hash"] for x in source]),
          "arrival_watermark_utc":max(bars[x]["arrival_watermark_utc"] for x in source),"proposal_arrival_utc":event["arrival_time_utc"],
          "shared_stream_targets":sorted(self.arms),"llm_called":False}
    def fanout(self,state,plans,proposal,event,proposal_hash,bbo_hash):
        inst=proposal["instrument"]; direction=proposal["direction"]; pip=.01 if inst.endswith("JPY") else .0001
        for arm,cost in sorted(self.arms.items()):
            bot=cost["inventory_controller"]=="BOT_ONLY"; action="ADD_WITHIN_CAP" if bot else "FREEZE"
            policy={"arm":arm,"controller":cost["inventory_controller"],"action":action,"worker_enabled":bot,
                    "currency_cap":1.0,"proposal_record_hash":proposal_hash,"hard_guards_mutable":False}
            policy_row=self.add(plans,"policy_receipts",policy,f"policy::{proposal['proposal_id']}::{arm}")
            order={"authority":"INTERNAL_EXPECTED_ONLY","external_submission_allowed":False,"proposal_id":proposal["proposal_id"],
              "proposal_record_hash":proposal_hash,"policy_receipt_hash":policy_row["record_hash"],"bbo_record_hash":bbo_hash,
              "arm":arm,"instrument":inst,"direction":direction,"action":action,"first_executable_event_id":event["event_id"]}
            order_row=self.add(plans,"expected_orders",order,f"expected::{proposal['proposal_id']}::{arm}"); state["counters"]["expected_orders"]+=1
            if not bot: continue
            slip=cost["slippage_pips"]*pip; price=float(event["ask"])+slip if direction>0 else float(event["bid"])-slip
            inv={"arm":arm,"instrument":inst,"direction":direction,"entry_price":price,"entry_event_id":event["event_id"],
                 "entry_time_utc":event["event_time_utc"],"proposal_id":proposal["proposal_id"],"proposal_record_hash":proposal_hash,
                 "policy_receipt_hash":policy_row["record_hash"],"expected_order_hash":order_row["record_hash"],
                 "bbo_record_hash":bbo_hash,"notional_fraction":1/7}
            state["inventories"][f"{arm}:{inst}"]=inv; self.add(plans,"virtual_fills",{"fill_kind":"VIRTUAL_ENTRY",**inv,"costs":cost}); state["counters"]["virtual_fills"]+=1
    def exit_if_due(self,state,plans,inst,bar):
        if bar["start_utc"][11:16]!="15:55": return
        pip=.01 if inst.endswith("JPY") else .0001
        for arm,cost in sorted(self.arms.items()):
            inv=state["inventories"].pop(f"{arm}:{inst}",None)
            if not inv: continue
            direction=inv["direction"]; slip=cost["slippage_pips"]*pip
            price=bar["bid_c"]-slip if direction>0 else bar["ask_c"]+slip
            gross=direction*(price/inv["entry_price"]-1); days=(parse_utc(bar["last_event_utc"])-parse_utc(inv["entry_time_utc"])).total_seconds()/86400
            net=gross-2*cost["commission_bps"]*1e-4-cost["financing_bps_day"]*1e-4*days; weighted=net/7
            state["realized_return"][arm]+=weighted
            self.add(plans,"virtual_fills",{"fill_kind":"VIRTUAL_EXIT","arm":arm,"instrument":inst,"proposal_id":inv["proposal_id"],
              "entry_fill_binding":inv["expected_order_hash"],"exit_price":price,"exit_bar_hash":bar["bar_record_hash"],
              "gross_return":gross,"net_return":net,"weighted_return":weighted,"costs":cost}); state["counters"]["virtual_exits"]+=1
    def accept(self,state,plans,event,adapter):
        state["counters"]["market_events_received"]+=1; code=self.quality(state,event,adapter)
        if code=="DUPLICATE_EVENT": state["counters"]["duplicate_events"]+=1; return
        if code!="OK":
            mapping={"SEQUENCE_GAP":"sequence_gaps","OUT_OF_ORDER_SEQUENCE":"out_of_order_events","OUT_OF_ORDER_TIME":"out_of_order_events","ARRIVAL_TIME_REGRESSION":"arrival_regressions","CLOCK_DRIFT":"clock_drift_events"}
            if code in mapping: state["counters"][mapping[code]]+=1
            state["counters"]["invalid_intervals"]+=1; state["counters"]["durable_halts"]+=1
            state.update(run_state="HALTED_INPUT" if code in ("MISSING_REQUIRED_FIELD","INVALID_FIELD_TYPE") else "HALTED_QUALITY",feed_blocked=True)
            state["block_reasons"].append(code); self.add(plans,"feed_quality",{"accepted":False,"code":code,"event_id":event.get("event_id")})
            self.add(plans,"control",{"halt_code":code,"run_state":state["run_state"]},f"halt::{code}::{event.get('event_id')}"); return
        raw={**event,"continuity":adapter.continuity,"lossless":adapter.lossless,"raw_payload_sha256":canonical_hash(event)}
        raw_row=self.add(plans,"raw_events",raw,f"event::{event['event_id']}")
        self.add(plans,"feed_quality",{"accepted":True,"code":"OK","event_id":event["event_id"],"raw_record_hash":raw_row["record_hash"]})
        state["counters"]["market_events_accepted"]+=1; inst=event["instrument"]; state["seen_events"][event["event_id"]]=canonical_hash(event)
        if adapter.continuity=="SEQUENCED": state["last_sequence"][inst]=int(event["source_sequence"])
        state["last_event_time"][inst]=event["event_time_utc"]; state["last_arrival_time"][inst]=event["arrival_time_utc"]
        state["last_quotes"][inst]={"bid":float(event["bid"]),"ask":float(event["ask"]),"event_time_utc":event["event_time_utc"],
          "arrival_time_utc":event["arrival_time_utc"],"event_id":event["event_id"],"bbo_record_hash":raw_row["record_hash"]}
        self.aggregate(state,plans,event); proposal=self.proposal(state,event)
        if proposal:
            proposal["input_checkpoint_root"]=canonical_hash({"raw_prefix_head":raw_row["record_hash"],"sealed_root":canonical_hash(state["sealed_periods"])})
            row=self.add(plans,"proposals",proposal,f"proposal::{proposal['proposal_id']}")
            state["proposal_ids"].append(proposal["proposal_id"]); state["proposals_by_id"][proposal["proposal_id"]]={**proposal,"record_hash":row["record_hash"]}
            state["counters"]["proposals"]+=1; state["counters"]["decisions"]+=1; self.fanout(state,plans,proposal,event,row["record_hash"],raw_row["record_hash"])
    def mark(self,state,plans,at):
        unreal={a:0.0 for a in self.arms}
        for inv in state["inventories"].values():
            quote=state["last_quotes"].get(inv["instrument"])
            if quote: unreal[inv["arm"]]+=inv["direction"]*((quote["bid"] if inv["direction"]>0 else quote["ask"])/inv["entry_price"]-1)/7
        state["unrealized_return"]=unreal
        self.add(plans,"pnl",{"at_utc":utc_text(at),"realized_return":state["realized_return"],"unrealized_return":unreal,
          "terminal_mtm_included":True,"open_inventory_count":len(state["inventories"])}); state["counters"]["pnl_records"]+=1
    def run(self,adapter,max_events=None,linger_seconds=0,fault_after=None):
        signal.signal(signal.SIGTERM,self.request_stop); signal.signal(signal.SIGINT,self.request_stop)
        with RuntimeLock(self.runtime_root,self.strategy_hash) as lock:
            if self.state["run_state"]!="RUNNING": self.write_status(); lock.heartbeat(self.status()); return self.status()
            try:
                source=adapter.source_bytes(); sid=str(adapter.path); previous=self.state["sources"].get(sid)
                if previous:
                    if len(source)<previous["size"] or hashlib.sha256(source[:previous["size"]]).hexdigest()!=previous["sha256"]:
                        self.durable_halt("INPUT_PREFIX_CHANGED","HALTED_INTEGRITY"); return self.status()
                    if len(source)==previous["size"]: self.write_status(); lock.heartbeat(self.status()); return self.status()
                state=copy.deepcopy(self.state); plans=self.plans(); raw_start=len(self.ledgers["raw_events"].rows)
                count=0
                for event in adapter.events():
                    if self.stop_requested or state["feed_blocked"]: break
                    self.accept(state,plans,event,adapter); count+=1
                    if max_events and count>=max_events: break
                if not state["feed_blocked"] and max_events is None:
                    digest=hashlib.sha256(source).hexdigest(); state["sources"][sid]={"size":len(source),"sha256":digest}
                    raw=plans["raw_events"]; manifest={"adapter_identity":adapter.feed_identity,"continuity":adapter.continuity,
                      "lossless":adapter.lossless,"source_size":len(source),"source_sha256":digest,
                      "raw_start_sequence":raw_start+1 if raw else raw_start,"raw_end_sequence":raw_start+len(raw),
                      "raw_transaction_hash":canonical_hash([r["record_hash"] for r in raw])}
                    self.add(plans,"batch_manifests",manifest,f"manifest::{digest}")
                at=max((parse_utc(v) for v in state["last_arrival_time"].values()),default=datetime.now(timezone.utc)); self.mark(state,plans,at)
                self.transact(state,plans,fault_after); self.write_status(); lock.heartbeat(self.status(),at)
                if linger_seconds:
                    end=time.monotonic()+linger_seconds
                    while not self.stop_requested and time.monotonic()<end: time.sleep(.02)
                return self.status()
            except (UnicodeDecodeError,json.JSONDecodeError,ValueError,TypeError) as exc:
                self.durable_halt("INVALID_INPUT","HALTED_INPUT",type(exc).__name__); return self.status()
    def request_stop(self,*_): self.stop_requested=True
    def tick(self,now):
        if self.state["run_state"]=="RUNNING" and self.state["last_arrival_time"]:
            latest=max(parse_utc(x) for x in self.state["last_arrival_time"].values())
            if (now-latest).total_seconds()>self.contract["continuity"]["max_silence_seconds"]:
                self.state["counters"]["silence_expiries"]+=1; self.durable_halt("HEARTBEAT_EXPIRED","HALTED_QUALITY")
        return self.status()
    def finalize_period(self,now):
        if not self.state["inventories"]:
            state=copy.deepcopy(self.state); state["period_state"]="SEALED"; self.transact(state,self.plans()); self.write_status(); return self.status()
        stale=[]
        for inv in self.state["inventories"].values():
            q=self.state["last_quotes"].get(inv["instrument"])
            if not q or (now-parse_utc(q["arrival_time_utc"])).total_seconds()>self.contract["continuity"]["max_quote_age_seconds"]: stale.append(inv["instrument"])
        state=copy.deepcopy(self.state); plans=self.plans(); code="STALE_TERMINAL_QUOTE" if stale else "OPEN_INVENTORY_AT_EOF"
        state.update(period_state="FAILED_UNPRICEABLE",run_state="HALTED_QUALITY",feed_blocked=True); state["block_reasons"].append(code); state["counters"]["durable_halts"]+=1
        self.add(plans,"control",{"halt_code":code,"open_inventory_count":len(state["inventories"])}); self.transact(state,plans); self.write_status(); return self.status()
    def ingest_policy_receipt(self,receipt):
        allowed={"proposal_id","proposal_record_hash","arm","worker_enabled","currency_cap","inventory_action","decision_timestamp_utc","arrival_timestamp_utc","model","input_hash","output_hash"}
        state=copy.deepcopy(self.state); plans=self.plans(); arm=receipt.get("arm","")
        valid=(set(receipt)<=allowed and arm in self.arms and self.arms[arm]["inventory_controller"]=="ACTUAL_LLM_POLICY_RECEIPT"
          and receipt.get("inventory_action") in {"ADD_WITHIN_CAP","FREEZE","UNWIND"} and isinstance(receipt.get("worker_enabled"),bool)
          and isinstance(receipt.get("currency_cap"),(int,float)) and not isinstance(receipt.get("currency_cap"),bool)
          and 0<=receipt.get("currency_cap")<=12 and receipt.get("proposal_id") in state["proposals_by_id"]
          and receipt.get("proposal_record_hash")==state["proposals_by_id"].get(receipt.get("proposal_id"),{}).get("record_hash"))
        self.add(plans,"policy_receipts",{"valid":valid,"receipt":receipt,"receipt_hash":canonical_hash(receipt)},f"external-policy::{canonical_hash(receipt)}")
        if not valid:
            if arm in state["arm_status"]: state["arm_status"][arm]="HALTED_POLICY"
            self.add(plans,"control",{"halt_code":"INVALID_LLM_POLICY","arm":arm})
        self.transact(state,plans); self.write_status(); return valid
    def status(self):
        return {"schema_version":2,"runtime_version":VERSION,"candidate_id":self.packet["candidate_id"],"strategy_hash":self.strategy_hash,
          "packet_hash":self.packet_hash,"runtime_contract_hash":self.contract_hash,"strategy_status":"RESEARCH_NOT_ADMITTED",
          "shadow_status":"OBSERVATION_AUTHORIZED","live_authority":"NONE","profit_proven":False,"monthly_2x_proven":False,
          "real_feed_binding":self.contract["feed_binding"]["status"],"real_feed_connected":self.contract["feed_binding"]["connected"],
          "real_feed_provider_candidate":self.contract["feed_binding"]["primary_provider_candidate"],
          "real_foreground_observation_enabled":self.contract["feed_binding"]["foreground_observation_enabled"],
          "credential_authority":self.contract["feed_binding"]["credential_authority"],
          "run_state":self.state["run_state"],"period_state":self.state["period_state"],"feed_blocked":self.state["feed_blocked"],
          "block_reasons":self.state["block_reasons"],"continuity":"SEQUENCED" if self.state["last_sequence"] else "HEARTBEAT_ONLY",
          "lossless":bool(self.state["last_sequence"]),"counters":self.state["counters"],"arm_status":self.state["arm_status"],
          "realized_return":self.state["realized_return"],"unrealized_return":self.state["unrealized_return"],
          "open_inventory_count":len(self.state["inventories"]),"ledger_heads":{n:x.last_hash for n,x in self.ledgers.items()},
          "checkpoint_root":self.checkpoint(self.state)["checkpoint_root"] if self.state_path.exists() else None}
    def write_status(self): atomic_json(self.status_path,self.status())


def generate_fixture(path,gap=False,truncate_at_noon=False):
    bases={"AUD_USD":.66,"EUR_USD":1.16,"GBP_USD":1.35,"NZD_USD":.59,"USD_CAD":1.37,"USD_CHF":.80,"USD_JPY":147.}
    rows=[]; start=datetime(2026,8,24,tzinfo=timezone.utc); last=720 if truncate_at_noon else 965
    for inst,base in bases.items():
        seq=0; pip=.01 if inst.endswith("JPY") else .0001
        for minute in range(0,last+1,5):
            seq+=1
            if gap and inst=="EUR_USD" and seq==50: continue
            drift=((minute//5)%7-3)*base*.00001 if minute<=355 else (0 if minute<480 else (base*.003*((minute-480)/235) if minute<=715 else base*(.003-.002*((minute-715)/245))))
            mid=base+drift; stamp=start+timedelta(minutes=minute)
            rows.append({"event_id":f"fixture::{inst}::{seq}","feed_identity":"LOCAL_FIXTURE_BBO_V1","instrument":inst,
              "source_sequence":seq,"event_time_utc":utc_text(stamp),"arrival_time_utc":utc_text(stamp+timedelta(seconds=.1)),
              "bid":round(mid-pip*.4,8),"ask":round(mid+pip*.4,8)})
    rows.sort(key=lambda x:(x["event_time_utc"],x["instrument"])); Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text("".join(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n" for r in rows),encoding="utf-8"); return len(rows)


def validate_heartbeat(path,now,max_age_seconds=90):
    try:
        payload=json.loads(secure_read(Path(path))); claimed=payload.pop("heartbeat_hash")
        if claimed!=canonical_hash(payload): return False,"HEARTBEAT_HASH_INVALID"
        if payload.get("run_state")!="RUNNING" or payload.get("feed_blocked") is not False: return False,"HEARTBEAT_NOT_RUNNING"
        age=(now-parse_utc(payload["beat_at_utc"])).total_seconds()
        if age < -2 or age>max_age_seconds: return False,"HEARTBEAT_STALE"
        if any(payload.get("counters",{}).get(k)!=0 for k in ZERO_CAPS): return False,"HEARTBEAT_CAPABILITY_COUNTER_NONZERO"
        return True,"HEARTBEAT_VALID"
    except Exception: return False,"HEARTBEAT_CORRUPT"


def main(argv=None):
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("generate-fixture"); p.add_argument("--output",type=Path,required=True); p.add_argument("--gap",action="store_true"); p.add_argument("--truncate-at-noon",action="store_true")
    p=sub.add_parser("run"); p.add_argument("--feed",type=Path,required=True); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--pace-seconds",type=float,default=0); p.add_argument("--linger-seconds",type=float,default=0); p.add_argument("--max-events",type=int); p.add_argument("--fault-after")
    p=sub.add_parser("tick"); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--now",required=True)
    p=sub.add_parser("finalize-period"); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--now",required=True)
    p=sub.add_parser("canonical-hash"); p.add_argument("--json",required=True)
    args=parser.parse_args(argv); root=Path(__file__).resolve().parent
    if args.command=="generate-fixture": print(json.dumps({"events":generate_fixture(args.output,args.gap,args.truncate_at_noon)},sort_keys=True)); return 0
    if args.command=="canonical-hash": print(canonical_hash(json.loads(args.json))); return 0
    try:
        runtime=ShadowRuntime(root,args.runtime_root)
        if args.command=="tick": status=runtime.tick(parse_utc(args.now))
        elif args.command=="finalize-period": status=runtime.finalize_period(parse_utc(args.now))
        else:
            status=runtime.run(JsonlFixtureAdapter(args.feed,args.pace_seconds),args.max_events,args.linger_seconds,args.fault_after)
            if status["open_inventory_count"] and not status["feed_blocked"] and args.max_events is None:
                latest=max(parse_utc(x) for x in runtime.state["last_arrival_time"].values()); status=runtime.finalize_period(latest)
        print(json.dumps(status,sort_keys=True)); return 2 if status["feed_blocked"] or status["period_state"]=="FAILED_UNPRICEABLE" else 0
    except InjectedCrash as exc: print(json.dumps({"injected_crash":str(exc)}),file=sys.stderr); return 99
    except Exception as exc: print(json.dumps({"error":type(exc).__name__}),file=sys.stderr); return 3


if __name__=="__main__": raise SystemExit(main())

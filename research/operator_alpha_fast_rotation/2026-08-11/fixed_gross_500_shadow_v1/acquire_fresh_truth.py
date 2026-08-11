#!/usr/bin/env python3
"""GET-only OANDA acquisition for the fixed-gross shadow experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.broker.oanda import OandaReadOnlyClient

ROOT = Path(__file__).resolve().parent
TRADES = (
    ("473162", "473180", "margin_closeout_1"), ("473183", "473186", "margin_closeout_2"),
    ("473189", "473191", "manual_win_1"), ("473193", "473195", "manual_win_2"),
    ("473197", "473199", "manual_win_3"), ("473201", "473204", "manual_win_4"),
    ("473212", "473218", "manual_win_5_overlapping_hedge_context"),
)
TOP = {"id","time","type","reason","instrument","units","price","pl","financing","commission","accountBalance","orderID","tradeID","tradeOpened","tradesClosed"}
LEG = {"price","tradeID","units","realizedPL","financing","halfSpreadCost","initialMarginRequired","homeConversionCost","plHomeConversionCost"}

def dt(v: str) -> datetime: return datetime.fromisoformat(v.replace("Z", "+00:00"))
def ot(v: datetime) -> str: return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
def clean_leg(v: dict[str, Any]) -> dict[str, Any]: return {k: v[k] for k in LEG if k in v}
def clean(v: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in TOP:
        if k not in v: continue
        if k == "tradeOpened" and isinstance(v[k], dict): out[k] = clean_leg(v[k])
        elif k == "tradesClosed" and isinstance(v[k], list): out[k] = [clean_leg(x) for x in v[k] if isinstance(x, dict)]
        else: out[k] = v[k]
    return out
def candle(v: dict[str, Any]) -> dict[str, Any] | None:
    if v.get("complete") is not True or not isinstance(v.get("bid"),dict) or not isinstance(v.get("ask"),dict): return None
    return {"time":v["time"],"complete":True,"volume":int(v.get("volume") or 0),"bid":{k:str(v["bid"][k]) for k in "ohlc"},"ask":{k:str(v["ask"][k]) for k in "ohlc"}}
def get_candles(c: OandaReadOnlyClient, pair: str, granularity: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
    p=c.get_json(f"/v3/instruments/{pair}/candles", {"granularity":granularity,"from":ot(start),"to":ot(end),"price":"BA","includeFirst":"true"})
    return [x for raw in p.get("candles",[]) if (x:=candle(raw)) is not None]
def dump(path: Path, v: Any) -> None: path.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def acquire(env_file: Path) -> dict[str, Any]:
    c=OandaReadOnlyClient(env_file=env_file); p=c.transactions_since_id("473160")
    by={str(v.get("id")):v for v in p.get("transactions",[]) if isinstance(v,dict)}
    ids={x for row in TRADES for x in row[:2]} | {"473207","473213","473215","473217"}
    missing=sorted(ids-{x for x in ids if x in by}, key=int)
    if missing: raise RuntimeError(f"missing required transaction IDs: {missing}")
    order_ids={str(by[x].get("orderID")) for x in ids if by[x].get("orderID")}
    transaction={"contract":"FIXED_GROSS_500_TRANSACTION_TRUTH_V1","read_only":True,"since_transaction_id":"473160","broker_last_transaction_id_at_acquisition":str(p.get("lastTransactionID") or ""),"trades":[{"entry_fill_id":a,"close_fill_id":b,"label":z} for a,b,z in TRADES],"transactions":[clean(by[x]) for x in sorted((ids|order_ids)&by.keys(),key=int)]}
    packets=[]
    for entry,_,_ in TRADES:
        v=by[entry]; when=dt(str(v["time"])); pair=str(v["instrument"])
        for gran,start,end in (("S5",when-timedelta(minutes=30),when+timedelta(minutes=65)),("M1",when-timedelta(hours=2),when+timedelta(minutes=65)),("H4",when-timedelta(days=21),when)):
            rows=get_candles(c,pair,gran,start,end)
            if not rows: raise RuntimeError(f"missing {gran} rows for {entry}")
            packets.append({"entry_fill_id":entry,"pair":pair,"granularity":gran,"rows":rows})
    open_trades=c.get_json(f"/v3/accounts/{c.account_id}/openTrades").get("trades",[])
    summary=c.get_json(f"/v3/accounts/{c.account_id}/summary").get("account",{})
    snapshot={"contract":"FIXED_GROSS_500_READONLY_FLAT_CHECK_V1","last_transaction_id":str(summary.get("lastTransactionID") or ""),"hedging_enabled":bool(summary.get("hedgingEnabled")),"open_trade_count":int(summary.get("openTradeCount") or 0),"open_trades":[{"id":str(x.get("id") or ""),"instrument":x.get("instrument"),"currentUnits":x.get("currentUnits"),"price":x.get("price"),"unrealizedPL":x.get("unrealizedPL")} for x in open_trades if isinstance(x,dict)]}
    for name,v in (("source_transactions_v1.json",transaction),("source_candles_v1.json",{"contract":"FIXED_GROSS_500_CANDLES_V1","complete_only":True,"price_component":"BID_ASK","packets":packets}),("broker_flat_check_v1.json",snapshot)): dump(ROOT/name,v)
    manifest={"contract":"FIXED_GROSS_500_SOURCE_MANIFEST_V1","permissions":{"live":False,"paper":False,"broker_mutation":False,"orders":False,"deploy":False,"broker_get_only":True},"files":{n:{"sha256":sha(ROOT/n),"bytes":(ROOT/n).stat().st_size} for n in ("source_transactions_v1.json","source_candles_v1.json","broker_flat_check_v1.json")}}
    dump(ROOT/"source_manifest_v1.json",manifest); return manifest
def main() -> int:
    a=argparse.ArgumentParser();a.add_argument("--env-file",type=Path,required=True);print(json.dumps(acquire(a.parse_args().env_file),ensure_ascii=False,indent=2));return 0
if __name__ == "__main__": raise SystemExit(main())
